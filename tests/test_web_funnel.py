"""/funnel endpoint wiring: validation, async accept, status polling, 404.

Skipped unless the optional ``web`` extra (FastAPI) is installed. The funnel job
itself is stubbed out so no universe fetch / LLM / graph runs — we're testing the
HTTP surface and the job store, not the pipeline (that's covered by the funnel
module tests).
"""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import tradingagents.web.server as server  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    # Neuter the background job so the executor does nothing heavy.
    monkeypatch.setattr(server, "_run_funnel_job", lambda job_id, req: None)
    server._JOBS.clear()
    return TestClient(server.app)


@pytest.mark.unit
def test_funnel_rejects_bad_date(client):
    assert client.post("/funnel", json={"date": "not-a-date"}).status_code == 422


@pytest.mark.unit
def test_funnel_accepts_and_creates_job(client):
    r = client.post("/funnel", json={"date": "2024-05-10", "top_screen": 5, "max_deep": 1})
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued" and "job_id" in body

    job = client.get(f"/funnel/{body['job_id']}")
    assert job.status_code == 200
    data = job.json()
    assert data["kind"] == "funnel"
    assert data["request"]["top_screen"] == 5


@pytest.mark.unit
def test_funnel_unknown_job_404(client):
    assert client.get("/funnel/does-not-exist").status_code == 404


@pytest.mark.unit
def test_funnel_validates_positive_cutoffs(client):
    # top_screen must be >= 1
    assert client.post("/funnel", json={"date": "2024-05-10", "top_screen": 0}).status_code == 422


@pytest.mark.unit
def test_funnel_page_served_on_get(client):
    r = client.get("/funnel")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Screening funnel" in r.text and "Run funnel" in r.text


@pytest.mark.unit
def test_index_links_to_funnel_page(client):
    assert 'href="/funnel"' in client.get("/").text


def _seed_run(base, date, picks):
    import json
    import os
    d = os.path.join(base, date)
    os.makedirs(d, exist_ok=True)
    rec = {"trade_date": date, "generated_at": date + "T00:00:00+00:00",
           "universe_size": 100, "screened": 10, "triaged": 3, "analyzed": len(picks),
           "picks": picks}
    with open(os.path.join(d, "run.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh)


@pytest.mark.unit
def test_list_runs_reads_archive_newest_first(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_funnel_runs_dir", lambda: str(tmp_path))
    _seed_run(tmp_path, "2024-05-09", [{"ticker": "A", "bucket": "BUY"}])
    _seed_run(tmp_path, "2024-05-10", [{"ticker": "B", "bucket": "HOLD"}, {"ticker": "C", "bucket": "BUY"}])
    r = client.get("/funnel/runs")
    assert r.status_code == 200
    runs = r.json()["runs"]
    assert [x["date"] for x in runs] == ["2024-05-10", "2024-05-09"]  # newest first
    assert runs[0]["analyzed"] == 2 and runs[0]["buys"] == 1


@pytest.mark.unit
def test_get_run_returns_full_record(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_funnel_runs_dir", lambda: str(tmp_path))
    _seed_run(tmp_path, "2024-05-10", [{"ticker": "B", "bucket": "BUY", "report": "WRITEUP"}])
    r = client.get("/funnel/runs/2024-05-10")
    assert r.status_code == 200
    assert r.json()["picks"][0]["report"] == "WRITEUP"


@pytest.mark.unit
def test_get_run_unknown_date_404(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_funnel_runs_dir", lambda: str(tmp_path))
    assert client.get("/funnel/runs/2099-01-01").status_code == 404


@pytest.mark.unit
def test_get_run_rejects_bad_date_and_traversal(client):
    assert client.get("/funnel/runs/not-a-date").status_code == 400
    assert client.get("/funnel/runs/../etc").status_code in (400, 404)


@pytest.mark.unit
def test_runs_route_not_shadowed_by_job_id(client, monkeypatch, tmp_path):
    # "runs" must hit the history endpoint, not be treated as a job_id (404 job).
    monkeypatch.setattr(server, "_funnel_runs_dir", lambda: str(tmp_path))
    assert client.get("/funnel/runs").status_code == 200


def _seed_theme_run(base, slug, date, label, picks):
    import json
    import os
    d = os.path.join(base, "themes", slug, date)
    os.makedirs(d, exist_ok=True)
    rec = {"trade_date": date, "label": label, "generated_at": date + "T00:00:00+00:00",
           "universe": ["AAA"], "universe_size": 1, "screened": 1, "triaged": 1,
           "analyzed": len(picks), "picks": picks}
    with open(os.path.join(d, "run.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh)


@pytest.mark.unit
def test_theme_accepted_and_stored_in_request(client):
    r = client.post("/funnel", json={"date": "2024-05-10", "theme": "data center"})
    assert r.status_code == 202
    job = client.get("/funnel/" + r.json()["job_id"]).json()
    assert job["request"]["theme"] == "data center"


@pytest.mark.unit
def test_list_themes(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_funnel_runs_dir", lambda: str(tmp_path))
    _seed_theme_run(tmp_path, "data-center", "2024-05-10", "data center",
                    [{"ticker": "NVDA", "bucket": "BUY"}])
    _seed_theme_run(tmp_path, "cybersecurity", "2024-05-09", "cybersecurity",
                    [{"ticker": "CRWD", "bucket": "HOLD"}])
    r = client.get("/funnel/themes")
    assert r.status_code == 200
    themes = r.json()["themes"]
    assert [t["slug"] for t in themes] == ["data-center", "cybersecurity"]  # newest first
    assert themes[0]["label"] == "data center" and themes[0]["buys"] == 1


@pytest.mark.unit
def test_get_theme_runs_and_record(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_funnel_runs_dir", lambda: str(tmp_path))
    _seed_theme_run(tmp_path, "data-center", "2024-05-10", "data center",
                    [{"ticker": "NVDA", "bucket": "BUY", "report": "WU"}])
    assert client.get("/funnel/themes/data-center").json()["runs"] == ["2024-05-10"]
    rec = client.get("/funnel/themes/data-center/2024-05-10").json()
    assert rec["label"] == "data center" and rec["picks"][0]["report"] == "WU"


@pytest.mark.unit
def test_theme_endpoints_validate_and_404(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_funnel_runs_dir", lambda: str(tmp_path))
    assert client.get("/funnel/themes/BAD_SLUG!").status_code == 400
    assert client.get("/funnel/themes/missing").status_code == 404
    assert client.get("/funnel/themes/data-center/not-a-date").status_code == 400
