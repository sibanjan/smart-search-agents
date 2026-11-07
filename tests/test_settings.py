"""The Settings page's API: checks, saving to config.yaml, and what each change triggers."""

import keyring
import pytest
import yaml
from keyring.backend import KeyringBackend

from smart_search_agents.daemon.scheduler import Scheduler
from smart_search_agents.daemon.service import create_app
from smart_search_agents.settings import Settings, load_settings


class MemoryKeyring(KeyringBackend):
    priority = 1

    def __init__(self):
        self.saved = {}

    def get_password(self, service, username):
        return self.saved.get((service, username))

    def set_password(self, service, username, password):
        self.saved[(service, username)] = password

    def delete_password(self, service, username):
        del self.saved[(service, username)]


@pytest.fixture
def vault():
    before = keyring.get_keyring()
    fake = MemoryKeyring()
    keyring.set_keyring(fake)
    yield fake
    keyring.set_keyring(before)


def values(client):
    return client.get("/api/settings").json()["values"]


def config_file(client):
    return yaml.safe_load(open(client.get("/api/settings").json()["config_file"], encoding="utf-8"))


def test_read_hides_what_cannot_be_edited(client, vault):
    data = client.get("/api/settings").json()
    assert data["values"]["agents.backend"] == "ollama" and data["claude_key_set"] is False
    assert "index_dir" not in data["values"] and "daemon.host" not in data["values"]


def test_bad_values_save_nothing(client):
    before = values(client)
    r = client.put("/api/settings", json={
        "default_limit": 7,  # valid, but the save is all-or-nothing
        "roots": [],
        "image_min_confidence": 2,
        "schedule.index": "not cron",
        "agents.backend": "gpt",
        "daemon.host": "0.0.0.0",
        "index_dir": "C:/elsewhere",
    })
    assert r.status_code == 400
    assert set(r.json()["detail"]) == {
        "roots", "image_min_confidence", "schedule.index", "agents.backend", "daemon.host", "index_dir",
    }
    assert values(client) == before


def test_chunk_overlap_must_fit_the_chunk_size(client):
    r = client.put("/api/settings", json={"chunk_size": 200, "chunk_overlap": 200})
    assert "chunk_overlap" in r.json()["detail"]


def test_live_setting_is_saved_and_applied_without_a_job(client, indexed):
    r = client.put("/api/settings", json={"default_limit": 7, "extensions": []}).json()["detail"]
    assert "extensions" in r  # still rejected: an empty list would index nothing
    r = client.put("/api/settings", json={"default_limit": 7}).json()
    assert r["changed"] == ["default_limit"] and r["jobs"] == [] and r["restart_required"] == []
    assert indexed.cfg.default_limit == 7
    assert config_file(client)["default_limit"] == 7


def test_unchanged_value_does_nothing(client):
    limit = values(client)["default_limit"]
    r = client.put("/api/settings", json={"default_limit": limit}).json()
    assert r["changed"] == [] and r["jobs"] == []


def test_new_folder_applies_and_queues_scans(client, indexed, jobs, tmp_path):
    new = tmp_path / "more"
    new.mkdir()
    (new / "tea.md").write_text("Oolong tea steeping notes.")
    roots = values(client)["roots"] + [str(new)]
    r = client.put("/api/settings", json={"roots": roots}).json()
    assert {j["kind"] for j in r["jobs"]} == {"index", "image_scan"}
    for job in r["jobs"]:
        assert jobs.wait(job["id"])["status"] == "done"
    assert client.get("/api/search", params={"q": "oolong"}).json()["results"][0]["name"] == "tea.md"


def test_extensions_are_normalised(client, jobs):
    r = client.put("/api/settings", json={"extensions": ["MD", ".txt", "md"]}).json()
    assert values(client)["extensions"] == [".md", ".txt"]
    assert [j["kind"] for j in r["jobs"]] == ["index"]


def test_category_change_reclassifies_without_a_scan(client, jobs):
    cats = {"reds": "something red", "blues": "something blue", "greens": "something green"}
    r = client.put("/api/settings", json={"image_categories": cats}).json()
    assert [j["kind"] for j in r["jobs"]] == ["image_classify"]
    jobs.wait(r["jobs"][0]["id"])
    assert "greens" in config_file(client)["image_categories"]


def test_removed_accepted_category_stays_removed(client, indexed, jobs):
    s = client.post("/api/suggestions", json={"name": "greens", "prompt": "something green"}).json()
    jobs.wait(client.post(f"/api/suggestions/{s['id']}/accept").json()["job"]["id"])
    client.put("/api/settings", json={"image_categories": {"reds": "something red"}})
    assert client.get("/api/suggestions", params={"status": "accepted"}).json()["suggestions"] == []


def test_model_change_waits_for_restart_and_warns(client, indexed):
    r = client.put("/api/settings", json={"model_name": "BAAI/bge-small-en-v1.5"}).json()
    assert r["restart_required"] == ["model_name"] and "clears" in r["warnings"][0]
    assert indexed.cfg.model_name != "BAAI/bge-small-en-v1.5"  # the running model is untouched
    data = client.get("/api/settings").json()
    assert data["values"]["model_name"] == "BAAI/bge-small-en-v1.5"
    assert data["restart_required"] == ["model_name"]


def test_remote_ollama_warns_that_excerpts_leave(client):
    r = client.put("/api/settings", json={"ollama_url": "http://10.0.0.5:11434"}).json()
    assert "10.0.0.5" in r["warnings"][0]
    r = client.put("/api/settings", json={"ollama_url": "http://localhost:11434"}).json()
    assert r["warnings"] == []


def test_saved_file_keeps_other_keys_and_backs_up_once(indexed, jobs, tmp_path):
    from fastapi.testclient import TestClient

    from smart_search_agents.daemon.curation import Curation

    path = tmp_path / "config.yaml"
    path.write_text("# my notes\nroots: [a]\nfuture_key: 1\nagents:\n  max_rounds: 4\n", encoding="utf-8")
    settings = Settings(core=indexed.cfg, config_file=path)
    app = create_app(indexed, jobs, Curation(indexed), settings, "t", allowed_hosts={"testserver"})
    with TestClient(app, headers={"Authorization": "Bearer t"}) as c:
        c.put("/api/settings", json={"agents.backend": "claude"})
        c.put("/api/settings", json={"default_limit": 9})
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["future_key"] == 1 and saved["default_limit"] == 9
    assert saved["agents"] == {"max_rounds": 4, "backend": "claude"}
    assert "# my notes" in path.with_name("config.yaml.bak").read_text(encoding="utf-8")
    # and the app reads back what it wrote
    assert load_settings(path).agents.backend == "claude"


def test_schedule_change_reloads_the_running_scheduler(indexed, jobs):
    from fastapi.testclient import TestClient

    from smart_search_agents.daemon.curation import Curation

    sched = Scheduler(jobs, {"index": "0 2 * * *", "image_scan": "0 3 * * SUN"})
    sched.start()
    try:
        settings = Settings(core=indexed.cfg, config_file=None)
        app = create_app(indexed, jobs, Curation(indexed), settings, "t", scheduler=sched,
                         allowed_hosts={"testserver"})
        with TestClient(app, headers={"Authorization": "Bearer t"}) as c:
            c.put("/api/settings", json={"schedule.index": "0 */6 * * *", "schedule.image_scan": ""})
            saved = yaml.safe_load(open(settings.config_file, encoding="utf-8"))
            runs = sched.next_runs()
            assert set(runs) == {"index"} and runs["index"]
            assert saved["schedule"] == {"index": "0 */6 * * *", "image_scan": ""}
            c.put("/api/settings", json={"schedule.image_scan": "0 3 * * *"})
            assert set(sched.next_runs()) == {"index", "image_scan"}
            saved = yaml.safe_load(open(settings.config_file, encoding="utf-8"))
            assert saved["schedule"]["index"] == "0 */6 * * *"
    finally:
        sched.stop()


def test_watcher_follows_new_folders(indexed, jobs, tmp_path):
    from smart_search_agents.daemon.watch import Watcher

    extra = tmp_path / "extra"
    extra.mkdir()
    w = Watcher(jobs, indexed.indexer)
    w.start()
    try:
        w.set_roots([extra])
        assert {e.watch.path for e in w.observer.emitters} == {str(extra)}
    finally:
        w.stop()


def test_claude_key_goes_to_the_vault_only(client, vault):
    assert client.put("/api/settings/claude_key", json={"key": "  "}).status_code == 400
    assert client.put("/api/settings/claude_key", json={"key": "sk-ant-test"}).json()["claude_key_set"]
    assert vault.saved[("SmartSearch", "anthropic_api_key")] == "sk-ant-test"
    data = client.get("/api/settings")
    assert data.json()["claude_key_set"] is True and "sk-ant-test" not in data.text
    assert client.delete("/api/settings/claude_key").json()["claude_key_set"] is False
    assert vault.saved == {}
    assert client.delete("/api/settings/claude_key").status_code == 200  # already gone


def test_page_has_settings_and_needs_token_for_api(client):
    from fastapi.testclient import TestClient

    assert TestClient(client.app).put("/api/settings", json={}).status_code == 401
    page = client.get("/").text
    assert 'id="vSettings"' in page and "SmartSearch" in page


def test_removing_a_folder_removes_its_files_and_images(client, jobs, tmp_path):
    assert client.get("/api/images/categories").json()["total"] == 5
    other = tmp_path / "other"
    other.mkdir()
    r = client.put("/api/settings", json={"roots": [str(other)]}).json()
    for job in r["jobs"]:
        assert jobs.wait(job["id"])["status"] == "done"
    assert client.get("/api/images/categories").json() == {"categories": {}, "total": 0}
    assert client.get("/api/status").json()["files"] == 0


def test_a_scan_of_one_folder_keeps_the_other_images(client, jobs, corpus):
    job = client.post("/api/jobs", json={"kind": "image_scan",
                                         "args": {"roots": [str(corpus / "notes")]}}).json()
    assert jobs.wait(job["id"])["status"] == "done"
    assert client.get("/api/images/categories").json()["total"] == 5
