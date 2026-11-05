import time

from smart_search_agents.daemon.jobs import JobQueue


def test_index_and_image_scan_jobs_complete(indexed, jobs):
    done = {j["kind"]: j for j in jobs.list(status="done")}
    assert done["index"]["report"]["added"] == 3
    assert done["image_scan"]["report"]["categories"] == {"reds": 3, "blues": 2}
    assert indexed.search("faithfulness")[0].name == "rag_eval.md"


def test_jobs_run_one_at_a_time(indexed, jobs):
    ids = [jobs.submit("index", {"force": True})["id"], jobs.submit("image_classify")["id"]]
    a, b = (jobs.wait(i) for i in ids)
    assert a["status"] == b["status"] == "done"
    assert b["started"] >= a["finished"]  # never two writers at once


def test_same_queued_job_is_coalesced(engine):
    q = JobQueue(engine)  # worker not started, so jobs stay queued
    assert q.submit("index")["id"] == q.submit("index")["id"]
    assert q.submit("index", {"force": True})["id"] != q.submit("index")["id"]


def test_watcher_paths_merge_into_one_job(engine):
    q = JobQueue(engine)
    first = q.submit("index_paths", {"paths": ["a.md", "b.md"]})
    second = q.submit("index_paths", {"paths": ["b.md", "c.md"]})
    assert first["id"] == second["id"]
    assert second["args"]["paths"] == ["a.md", "b.md", "c.md"]


def test_index_paths_indexes_and_removes(indexed, jobs, corpus):
    new = corpus / "notes" / "new.md"
    new.write_text("quantum chromodynamics lecture")
    (corpus / "notes" / "pasta.md").unlink()
    job = jobs.wait(jobs.submit("index_paths", {"paths": [str(new), str(corpus / "notes" / "pasta.md")]})["id"])
    assert job["report"] == {"indexed": 1, "removed": 1, "skipped": 0}
    assert indexed.search("chromodynamics")[0].name == "new.md"


def test_cancel_queued_and_running(engine):
    q = JobQueue(engine)

    def slow(args, ctx):
        for _ in range(200):
            ctx.check()
            time.sleep(0.01)
        return {}

    q.handlers["index"] = slow
    running = q.submit("index")
    queued = q.submit("image_classify")
    q.start()
    time.sleep(0.2)
    assert q.cancel(queued["id"])["status"] == "cancelled"
    q.cancel(running["id"])
    assert q.wait(running["id"])["status"] == "cancelled"
    q.stop()


def test_failed_job_records_error_and_worker_survives(engine, jobs):
    def boom(args, ctx):
        raise RuntimeError("disk on fire")

    jobs.handlers["image_classify"] = boom
    failed = jobs.wait(jobs.submit("image_classify")["id"])
    assert failed["status"] == "failed" and "disk on fire" in failed["error"]
    assert jobs.wait(jobs.submit("index")["id"])["status"] == "done"


def test_restart_marks_orphaned_running_jobs_failed(engine):
    q = JobQueue(engine)
    job = q.submit("index")
    with engine.store.conn:
        engine.store.conn.execute("UPDATE jobs SET status='running' WHERE id=?", (job["id"],))
    q2 = JobQueue(engine)
    assert q2.get(job["id"])["status"] == "failed"


def test_unknown_kind_rejected(engine):
    import pytest

    with pytest.raises(ValueError):
        JobQueue(engine).submit("format_disk")
