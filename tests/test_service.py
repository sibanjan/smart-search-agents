from fastapi.testclient import TestClient


def test_token_and_host_are_required(client):
    bare = TestClient(client.app)
    assert bare.get("/api/status").status_code == 401
    assert bare.get("/api/status", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert bare.get("/api/status", headers={"Host": "evil.example"}).status_code == 403
    # the web UI page hands its own requests the token as a same-site cookie
    assert bare.get("/").status_code == 200
    assert bare.get("/api/status").status_code == 200


def test_search_and_chunks(client):
    hits = client.get("/api/search", params={"q": "faithfulness relevance"}).json()["results"]
    assert hits[0]["name"] == "rag_eval.md"
    chunks = client.get("/api/chunks", params={"path": hits[0]["path"], "query": "faithfulness"}).json()
    assert "faithfulness" in chunks["chunks"][0]["text"]
    assert client.get("/api/chunks", params={"path": "C:/not/indexed.txt"}).status_code == 404


def test_jobs_api(client):
    job = client.post("/api/jobs", json={"kind": "image_classify"}).json()
    assert job["kind"] == "image_classify" and job["trigger"] == "api"
    assert client.get(f"/api/jobs/{job['id']}").status_code == 200
    assert client.post("/api/jobs", json={"kind": "nope"}).status_code == 400
    assert client.post("/api/reindex").json()["job"]["kind"] == "index"


def test_readonly_sql(client):
    ok = client.post("/api/query", json={"sql": "SELECT category, COUNT(*) n FROM image_category_view "
                                                 "GROUP BY category ORDER BY category"}).json()
    assert ok["columns"] == ["category", "n"] and ok["rows"] == [["blues", 2], ["reds", 3]]
    vec = client.post("/api/query", json={"sql": "SELECT vec FROM images LIMIT 1"}).json()
    assert vec["rows"][0][0].endswith("bytes>")
    for bad in ("DELETE FROM files", "SELECT 1; DROP TABLE files", "PRAGMA writable_schema=1",
                "WITH x AS (SELECT 1) DELETE FROM files"):
        assert client.post("/api/query", json={"sql": bad}).status_code == 400, bad
    assert client.get("/api/status").json()["files"] >= 3  # nothing was deleted


def test_image_detail_and_config(client):
    path = client.get("/api/images", params={"category": "reds"}).json()["images"][0]["path"]
    detail = client.get("/api/images/detail", params={"path": path}).json()
    assert detail["labels"][0]["category"] == "reds" and detail["labels"][0]["primary"]
    cfg = client.get("/api/config").json()
    assert cfg["image_categories"] == {"reds": "something red", "blues": "something blue"}
    assert "token" not in str(cfg).lower()


def test_preview_then_accept_suggestion(client, jobs):
    preview = client.post("/api/images/preview_category",
                          json={"name": "crimson", "prompt": "deep red"}).json()
    assert preview["total_images"] == 5
    assert set(preview["from"]) <= {"reds", "blues"}  # a dry run: links are unchanged
    assert client.get("/api/images/categories").json()["categories"] == {"reds": 3, "blues": 2}
    s = client.post("/api/suggestions", json={"name": "greens", "prompt": "something green",
                                              "rationale": "test"}).json()
    assert s["status"] == "proposed" and s["preview"]["would_join"] == 0
    r = client.post(f"/api/suggestions/{s['id']}/accept").json()
    assert r["suggestion"]["status"] == "accepted"
    assert jobs.wait(r["job"]["id"])["status"] == "done"
    assert "greens" in client.get("/api/config").json()["image_categories"]
    assert "greens:" in client.get("/api/suggestions/export").json()["yaml"]
    assert client.post(f"/api/suggestions/{s['id']}/maybe").status_code == 400


def test_daemon_status(client):
    s = client.get("/api/daemon/status").json()
    assert s["images"] == 5 and s["last_done"]["index"]["id"]


def test_page_is_never_cached(client):
    # it carries the session cookie; a cached copy would load with every API call failing
    r = client.get("/")
    assert r.headers["cache-control"] == "no-store" and "ssa_session" in r.headers["set-cookie"]
