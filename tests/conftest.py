import hashlib

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from smartsearch.config import Config

from smart_search_agents.daemon.curation import Curation
from smart_search_agents.daemon.db import make_engine
from smart_search_agents.daemon.jobs import JobQueue
from smart_search_agents.daemon.service import create_app
from smart_search_agents.settings import Settings

TOKEN = "test-token"
COLORS = {"red": (220, 20, 20), "blue": (20, 20, 220), "green": (20, 200, 20)}


class FakeClip:
    """Images -> mean colour; texts naming a colour -> that colour, else a stable random vector."""

    name = "fake-clip"

    def encode_images(self, images):
        return _norm(np.array([np.asarray(im, dtype=np.float32).mean(axis=(0, 1)) for im in images]))

    def encode_texts(self, texts):
        out = []
        for t in texts:
            hit = next((rgb for c, rgb in COLORS.items() if c in t), None)
            if hit is None:
                seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
                hit = np.random.default_rng(seed).normal(size=3)
            out.append(hit)
        return _norm(np.array(out, dtype=np.float32))


def _norm(m):
    return (m / np.linalg.norm(m, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture
def corpus(tmp_path):
    docs = tmp_path / "docs"
    (docs / "notes").mkdir(parents=True)
    (docs / "notes" / "rag_eval.md").write_text(
        "# RAG evaluation\n\nMeasure faithfulness and answer relevance with a held-out question set."
    )
    (docs / "notes" / "pasta.md").write_text("Carbonara: spaghetti, eggs, pecorino, guanciale.")
    (docs / "budget.csv").write_text("category,amount\nrent,1800\n")
    pics = docs / "pics"
    pics.mkdir()
    for color, n in (("red", 3), ("blue", 2)):
        for i in range(n):
            Image.new("RGB", (64, 64), COLORS[color]).save(pics / f"{color}_{i}.png")
    return docs


@pytest.fixture
def engine(tmp_path, corpus):
    cfg = Config(
        roots=[str(corpus)],
        index_dir=str(tmp_path / "index"),
        embedder="hash",
        image_min_size_kb=0,
        image_categories={"reds": "something red", "blues": "something blue"},
        image_min_confidence=0.6,
    )
    return make_engine(cfg, FakeClip)


@pytest.fixture
def jobs(engine):
    q = JobQueue(engine)
    q.start()
    yield q
    q.stop()


@pytest.fixture
def indexed(engine, jobs):
    for kind in ("index", "image_scan"):
        job = jobs.wait(jobs.submit(kind, trigger="cli")["id"])
        assert job["status"] == "done", job
    return engine


@pytest.fixture
def client(indexed, jobs):
    settings = Settings(core=indexed.cfg, config_file=None)
    app = create_app(indexed, jobs, Curation(indexed), settings, TOKEN,
                     allowed_hosts={"testserver", "127.0.0.1"})
    with TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        yield c
