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
