from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import FakeEngine, make_tone


@pytest.fixture
def client(isolated_config):
    app = create_app(engine=FakeEngine(), allowed_hosts=["*"])
    with TestClient(app) as test_client:
        yield test_client


def upload_voice(client: TestClient, tmp_path: Path, seconds: float = 12.0) -> str:
    source = make_tone(tmp_path / f"voice_{seconds}.wav", seconds=seconds)
    with source.open("rb") as handle:
        response = client.post(
            "/api/voices",
            data={"name": "测试音色", "transcript": "这是一段测试转写。"},
            files={"reference_audio": ("reference.wav", handle, "audio/wav")},
        )
    assert response.status_code == 201, response.text
    return response.json()["voice_id"]


def test_health(client: TestClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["queue_capacity"] >= 4


def test_voice_upload_tts_audio_history_delete(client: TestClient, isolated_config):
    voice_id = upload_voice(client, isolated_config)

    voices = client.get("/api/voices").json()["voices"]
    assert any(voice["voice_id"] == voice_id for voice in voices)

    response = client.post(
        "/api/tts",
        json={
            "text": "今天早上的天气有点凉。",
            "voice_id": voice_id,
            "quality": "fast",
            "postprocess": "clean",
            "client_request_id": "test-request-0001",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["request_id"] == "test-request-0001"
    assert body["flow_steps"] == 4
    assert body["postprocess"] == "clean"
    assert body["duration_sec"] > 0
    assert body["audio_url"].startswith("/api/audio/")

    audio = client.get(body["audio_url"])
    assert audio.status_code == 200
    assert audio.headers["content-type"].startswith("audio/wav")
    assert len(audio.content) > 100

    history = client.get("/api/history?limit=5")
    assert history.status_code == 200
    items = history.json()["items"]
    assert any(item["request_id"] == "test-request-0001" for item in items)

    assert client.delete(f"/api/voices/{voice_id}").status_code == 204
    assert client.get("/api/voices").json()["voices"] == []


def test_default_direct_endpoint(client: TestClient, isolated_config):
    source = make_tone(isolated_config / "default_reference.wav", seconds=12.0)
    client.app.state.voices.create(
        "default",
        source,
        "这是一段测试转写。",
        voice_id="default",
    )
    response = client.post(
        "/api/tts/default",
        json={"text": "测试 default 直连接口。", "quality": "fast", "postprocess": "clean"},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.headers["x-voice"] == "default"
    assert len(response.content) > 100


def test_long_job_progress_and_audio(client: TestClient, isolated_config):
    voice_id = upload_voice(client, isolated_config)
    response = client.post(
        "/api/tts/jobs",
        json={
            "text": "无" * 1000,
            "voice_id": voice_id,
            "model": "ming",
            "quality": "fast",
            "postprocess": "clean",
            "client_request_id": "test-long-0001",
        },
    )
    assert response.status_code == 202, response.text
    body = response.json()
    job_id = body["job_id"]
    assert body["segments_total"] == 13

    status = None
    for _ in range(150):
        status = client.get(f"/api/tts/jobs/{job_id}").json()
        if status["status"] in {"done", "error", "cancelled", "interrupted"}:
            break
        time.sleep(0.1)
    assert status is not None
    assert status["status"] == "done", status
    assert status["chars_total"] >= 1000
    assert status["chars_done"] == status["chars_total"]
    assert status["progress"] == 1.0
    assert status["audio_url"]
    audio = client.get(status["audio_url"])
    assert audio.status_code == 200
    assert len(audio.content) > 100
    history = client.get("/api/history?limit=5").json()["items"]
    assert any(item["request_id"] == job_id for item in history)


def test_control_conflict(client: TestClient, isolated_config):
    voice_id = upload_voice(client, isolated_config)
    response = client.post(
        "/api/tts",
        json={
            "text": "测试",
            "voice_id": voice_id,
            "emotion": "高兴",
            "speed": "快速",
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_CONTROL_COMBINATION"


def test_long_text(client: TestClient, isolated_config):
    voice_id = upload_voice(client, isolated_config)
    response = client.post(
        "/api/tts",
        json={"text": "无" * 501, "voice_id": voice_id},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "TEXT_TOO_LONG"


def test_missing_voice(client: TestClient):
    response = client.post(
        "/api/tts",
        json={"text": "测试", "voice_id": "vmissing123"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "VOICE_NOT_FOUND"


def test_invalid_audio(client: TestClient, isolated_config):
    source = make_tone(isolated_config / "short.wav", seconds=2.0)
    with source.open("rb") as handle:
        response = client.post(
            "/api/voices",
            data={"name": "太短", "transcript": "太短。"},
            files={"reference_audio": ("short.wav", handle, "audio/wav")},
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_AUDIO"


def test_path_traversal(client: TestClient):
    response = client.get("/api/audio/20260101/..%2Fsecret.wav")
    assert response.status_code in (400, 404)


def test_token_required(isolated_config):
    app = create_app(engine=FakeEngine(), api_token="secret", allowed_hosts=["*"])
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/voices").status_code == 401
        response = client.get("/api/voices", headers={"Authorization": "Bearer secret"})
        assert response.status_code == 200


def test_invalid_model_rejected(client: TestClient, isolated_config):
    voice_id = upload_voice(client, isolated_config)
    response = client.post(
        "/api/tts",
        json={"text": "测试", "voice_id": voice_id, "model": "bad-model"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"
