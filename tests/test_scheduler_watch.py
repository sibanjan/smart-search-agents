from datetime import datetime, timezone

from smart_search_agents.daemon.jobs import JobQueue
from smart_search_agents.daemon.scheduler import Scheduler
from smart_search_agents.daemon.watch import _Handler


def test_interval_from_cron(engine):
    s = Scheduler(JobQueue(engine), {"index": "0 2 * * *", "image_scan": "0 3 * * SUN"})
    now = datetime(2026, 11, 5, 12, tzinfo=timezone.utc)
    assert Scheduler.interval(s.triggers["index"], now).total_seconds() == 86400
    assert Scheduler.interval(s.triggers["image_scan"], now).days == 7


def test_catch_up_queues_only_stale_jobs(engine):
    q = JobQueue(engine)
    s = Scheduler(q, {"index": "0 2 * * *", "image_scan": "0 3 * * SUN", "bogus": "* * * * *"})
    assert "bogus" not in s.triggers
    assert sorted(s.catch_up()) == ["image_scan", "index"]  # never ran
    q.start()
    for j in q.list(status="queued"):
        q.wait(j["id"])
    q.stop()
    assert s.catch_up() == []  # both ran just now


def test_watcher_queues_one_job_for_candidate_paths(engine, corpus):
    q = JobQueue(engine)
    h = _Handler(q, engine.indexer, delay=3600)  # flush manually
    for p in ("notes/a.md", "notes/a.md", "notes/b.txt", "node_modules/x.md", "photo.png"):
        h._add(str(corpus / p))
    h.flush()
    h.stop()
    (job,) = q.list(status="queued")
    assert job["kind"] == "index_paths" and job["trigger"] == "watcher"
    assert [p.replace("\\", "/").split("/docs/")[1] for p in job["args"]["paths"]] == [
        "notes/a.md", "notes/b.txt"]
