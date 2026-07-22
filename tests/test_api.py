"""Tests for the FastAPI backend (api.py).

Uses the tiny, already-committed ``web_model/gpt.pt`` bundle (809K params) as
the model fixture -- fast and offline, no training or network access happens
in these tests.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import api


def test_info_reports_ready_model():
    with TestClient(api.app) as client:
        resp = client.get("/api/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert body["params"] > 0
    assert body["vocab"] > 0
    assert body["block_size"] > 0


def test_generate_returns_text_for_prompt():
    with TestClient(api.app) as client:
        resp = client.post("/api/generate", json={
            "prompt": "ROMEO:", "max_new_tokens": 10, "temperature": 0.8, "top_k": 40,
        })
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert isinstance(body["text"], str) and len(body["text"]) > 0
    assert body["prompt"].startswith("ROMEO:") or body["prompt"] == "\n"


def test_generate_clamps_out_of_range_params():
    with TestClient(api.app) as client:
        resp = client.post("/api/generate", json={
            "prompt": "test", "max_new_tokens": 999999, "temperature": 0.0, "top_k": 0,
        })
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_cors_default_allows_local_dev_origins():
    with TestClient(api.app) as client:
        resp = client.get("/api/info", headers={"Origin": "http://localhost:5173"})
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
