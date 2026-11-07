"""Category proposals for images: preview a prompt without writing, and keep suggestions
for a person to accept or reject. Nothing here edits config.yaml or moves files.

Accepted categories are stored in the DB and added to the running config on top of
`image_categories`; `ssa suggestions export` prints them as YAML to paste into config.
"""

from __future__ import annotations

import json
import time

import numpy as np
from smartsearch.engine import Engine
from smartsearch.images import UNCATEGORIZED

SCHEMA = """
CREATE TABLE IF NOT EXISTS category_suggestions (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    prompt TEXT NOT NULL,
    rationale TEXT,
    evidence TEXT NOT NULL DEFAULT '[]',  -- image paths the proposer looked at
    preview TEXT,                          -- preview_category() result when proposed
    status TEXT NOT NULL,                  -- proposed | accepted | rejected | removed
    created REAL NOT NULL,
    decided REAL
);
"""


class Curation:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.store = engine.store
        with self.store.lock, self.conn:
            self.conn.executescript(SCHEMA)
        self.apply_accepted()

    @property
    def conn(self):
        return self.store.conn

    # ---- preview (read-only) ---------------------------------------------
    def preview(self, name: str, prompt: str, examples: int = 12) -> dict:
        """Score every image against the current categories plus the new one and report
        which images would get it as their primary category. Writes nothing."""
        cat = self.engine.images
        rows = self.conn.execute("SELECT id, path, vec FROM images").fetchall()
        if not rows:
            return {"name": name, "would_join": 0, "from": {}, "examples": [], "total_images": 0}
        x = np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
        cats = {k: v for k, v in self.engine.cfg.image_categories.items() if k != name}
        names = list(cats) + [name]
        t = cat.encoder.encode_texts(list(cats.values()) + [prompt])
        logits = 100.0 * (x @ t.T)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        new = len(names) - 1
        top = probs.argmax(axis=1)
        joins = np.flatnonzero((top == new) & (probs[:, new] >= self.engine.cfg.image_min_confidence))

        primary = {
            r["image_id"]: r["category"]
            for r in self.conn.execute(
                "SELECT image_id, category FROM image_labels WHERE is_primary=1"
            )
        }
        moved_from: dict[str, int] = {}
        for i in joins:
            old = primary.get(rows[i]["id"], UNCATEGORIZED)
            moved_from[old] = moved_from.get(old, 0) + 1
        best = joins[np.argsort(-probs[joins, new])][:examples]
        return {
            "name": name,
            "prompt": prompt,
            "would_join": int(len(joins)),
            "from": dict(sorted(moved_from.items(), key=lambda kv: -kv[1])),
            "mean_score": round(float(probs[joins, new].mean()), 3) if len(joins) else 0.0,
            "examples": [
                {"path": rows[i]["path"], "score": round(float(probs[i, new]), 3)} for i in best
            ],
            "total_images": len(rows),
        }

    # ---- suggestions -----------------------------------------------------
    def suggest(
        self, name: str, prompt: str, evidence: list[str] | None = None, rationale: str = ""
    ) -> dict:
        name = name.strip()
        if not name or not prompt.strip():
            raise ValueError("name and prompt are required")
        preview = self.preview(name, prompt) if self._has_images() else None
        with self.store.lock, self.conn:
            cur = self.conn.execute(
                "INSERT INTO category_suggestions(name,prompt,rationale,evidence,preview,status,created) "
                "VALUES(?,?,?,?,?,?,?)",
                (name, prompt.strip(), rationale, json.dumps(evidence or []),
                 json.dumps(preview) if preview else None, "proposed", time.time()),
            )
        return self.get(cur.lastrowid)

    def get(self, sid: int) -> dict | None:
        r = self.conn.execute("SELECT * FROM category_suggestions WHERE id=?", (sid,)).fetchone()
        return _row(r) if r else None

    def list(self, status: str | None = None) -> list[dict]:
        sql, params = "SELECT * FROM category_suggestions", []
        if status:
            sql += " WHERE status=?"
            params.append(status)
        return [_row(r) for r in self.conn.execute(sql + " ORDER BY id DESC", params)]

    def decide(self, sid: int, accept: bool) -> dict:
        s = self.get(sid)
        if s is None:
            raise KeyError(sid)
        if s["status"] != "proposed":
            return s
        with self.store.lock, self.conn:
            self.conn.execute(
                "UPDATE category_suggestions SET status=?, decided=? WHERE id=?",
                ("accepted" if accept else "rejected", time.time(), sid),
            )
        if accept:
            self.engine.cfg.image_categories[s["name"]] = s["prompt"]
        return self.get(sid)

    def retire(self, names: set[str]) -> None:
        """The user removed these categories in Settings: stop re-adding them at start-up."""
        with self.store.lock, self.conn:
            for name in names:
                self.conn.execute(
                    "UPDATE category_suggestions SET status='removed' "
                    "WHERE name=? AND status='accepted'",
                    (name,),
                )

    def apply_accepted(self) -> None:
        for s in self.list("accepted"):
            self.engine.cfg.image_categories.setdefault(s["name"], s["prompt"])

    def export_yaml(self) -> str:
        lines = ["image_categories:"]
        lines += [f'  {k}: "{v}"' for k, v in self.engine.cfg.image_categories.items()]
        return "\n".join(lines) + "\n"

    def _has_images(self) -> bool:
        return self.conn.execute("SELECT 1 FROM images LIMIT 1").fetchone() is not None


def _row(r) -> dict:
    d = dict(r)
    d["evidence"] = json.loads(d["evidence"] or "[]")
    d["preview"] = json.loads(d["preview"]) if d.get("preview") else None
    return d
