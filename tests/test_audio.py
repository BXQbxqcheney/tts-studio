from __future__ import annotations

from pathlib import Path

import pytest

from app.services import audio
from tests.conftest import make_tone


def test_convert_reference(tmp_path: Path):
    source = make_tone(tmp_path / "source.wav", seconds=12.0)
    target = tmp_path / "reference.wav"
    duration = audio.convert_reference(source, target)
    assert target.is_file()
    assert 10.0 <= duration <= 13.0
    assert audio.read_wav(target)[1] == 44100


def test_postprocess_modes(tmp_path: Path):
    source = tmp_path / "source.wav"
    make_tone(source, seconds=2.0)
    for mode in ("clean", "natural", "warm"):
        target = tmp_path / f"{mode}.wav"
        audio.postprocess(source, target, mode)
        assert target.is_file()
        peak, rms = audio.peak_and_rms(target)
        assert peak <= -0.5
        assert rms < -5.0


def test_select_reference(tmp_path: Path):
    clips = []
    for index in range(4):
        path = make_tone(tmp_path / f"{index}.wav", seconds=4.0)
        clips.append((path, f"第{index}句。"))
    reference = tmp_path / "reference.wav"
    transcript = tmp_path / "reference.txt"
    duration = audio.select_reference(clips, reference, transcript, min_sec=10.0, max_sec=15.0)
    assert 10.0 <= duration <= 15.5
    assert reference.is_file()
    assert transcript.read_text(encoding="utf-8").strip()


def test_reference_too_short(tmp_path: Path):
    clips = [(make_tone(tmp_path / "short.wav", seconds=3.0), "太短。")]
    with pytest.raises(Exception):
        audio.select_reference(clips, tmp_path / "out.wav", tmp_path / "out.txt")


def test_concat_with_gap_infers_32k_rate(tmp_path: Path):
    paths = []
    for index in range(2):
        path = tmp_path / f"seg_{index}.wav"
        import soundfile as sf
        import numpy as np

        sf.write(str(path), np.zeros(32000, dtype=np.float32), 32000, subtype="PCM_16")
        paths.append(path)
    target = tmp_path / "joined.wav"
    duration = audio.concat_with_gap(paths, target, gap_sec=0.2)
    data, rate = audio.read_wav(target)
    assert rate == 32000
    assert 2.1 <= duration <= 2.3
    assert data.size == int(32000 * duration)
