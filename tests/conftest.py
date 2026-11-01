from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.engines.base import SegmentResult  # noqa: E402


class FakeEngine:
    name = "fake"
    sample_rate = 44100

    def __init__(self) -> None:
        self._ready = True
        self._error: str | None = None

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> str | None:
        return self._error

    def load(self) -> None:
        self._ready = True

    def synthesize(self, text, voice_wav, voice_text, task, controls, flow_steps, model=None, mode=None):
        sample_rate = 44100
        length = int(sample_rate * 0.5)
        time = np.arange(length, dtype=np.float32) / sample_rate
        waveform = (0.12 * np.sin(2 * np.pi * 220.0 * time)).astype(np.float32)
        return SegmentResult(waveform=waveform, sample_rate=sample_rate, infer_sec=0.01, vram_peak_mb=0.0)

    def clear_prompt_cache(self) -> None:
        return None

    def status(self) -> dict:
        return {
            "model_loaded": True,
            "loading": False,
            "device": "cpu",
            "gpu": "fake",
            "sample_rate": 44100,
            "vram_allocated_mb": 0.0,
            "vram_peak_mb": 0.0,
            "error": None,
        }


def make_tone(path: Path, seconds: float = 12.0, sample_rate: int = 44100) -> Path:
    length = int(seconds * sample_rate)
    time = np.arange(length, dtype=np.float32) / sample_rate
    envelope = np.minimum(1.0, np.minimum(time * 20.0, (seconds - time) * 20.0))
    waveform = (0.15 * envelope * np.sin(2 * np.pi * 220.0 * time)).astype(np.float32)
    sf.write(str(path), waveform, sample_rate, subtype="PCM_16")
    return path


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VOICES_DIR", tmp_path / "voices")
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(config, "HISTORY_FILE", tmp_path / "outputs" / "history.jsonl")
    monkeypatch.setattr(config, "TMP_DIR", tmp_path / "tmp")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    config.ensure_directories()
    return tmp_path
