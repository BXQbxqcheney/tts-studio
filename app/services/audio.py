"""音频转换、参考段拼接、响度归一化与自然人声后处理。"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from app import config
from app.errors import AppError

logger = logging.getLogger("tts_studio.audio")
SAMPLE_RATE = 44100


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise AppError("FFMPEG_MISSING", "系统未找到 ffmpeg", 500)
    return path


def _ffprobe() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise AppError("FFMPEG_MISSING", "系统未找到 ffprobe", 500)
    return path


def _run(command: list[str], error_code: str = "INVALID_AUDIO") -> None:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        logger.error("ffmpeg failed: %s\n%s", " ".join(command), result.stderr[-2000:])
        raise AppError(error_code, "音频处理失败，请检查文件格式", 400)


def duration_sec(path: Path) -> float:
    command = [
        _ffprobe(),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        logger.error("ffprobe failed: %s", result.stderr[-1000:])
        raise AppError("INVALID_AUDIO", "无法读取音频信息", 400)
    try:
        value = float(result.stdout.strip())
    except ValueError as exc:
        raise AppError("INVALID_AUDIO", "音频时长无效", 400) from exc
    if not math.isfinite(value) or value <= 0:
        raise AppError("INVALID_AUDIO", "音频时长无效", 400)
    return value


def convert_reference(src: Path, dst: Path) -> float:
    """转 44.1kHz 单声道 PCM16，去首尾静音并做响度归一化。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    filters = (
        "silenceremove=start_periods=1:start_threshold=-45dB:start_duration=0.1,"
        "areverse,"
        "silenceremove=start_periods=1:start_threshold=-45dB:start_duration=0.1,"
        "areverse,"
        "loudnorm=I=-18:TP=-2:LRA=11"
    )
    _run(
        [
            _ffmpeg(),
            "-y",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            "-af",
            filters,
            str(dst),
        ]
    )
    return duration_sec(dst)


def trim_to(src: Path, dst: Path, max_sec: float) -> None:
    _run(
        [
            _ffmpeg(),
            "-y",
            "-i",
            str(src),
            "-t",
            f"{max_sec:.3f}",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
    )


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32), int(sample_rate)


def write_pcm16(path: Path, data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(data, dtype=np.float32), sample_rate, subtype="PCM_16")


def concat_with_gap(paths: list[Path], dst: Path, gap_sec: float = 0.2, sample_rate: int | None = None) -> float:
    chunks: list[np.ndarray] = []
    rates: set[int] = set()
    for path in paths:
        data, rate = read_wav(path)
        rates.add(rate)
        chunks.append(data)
        chunks.append(np.zeros(int(rate * gap_sec), dtype=np.float32))
    if len(rates) > 1:
        raise AppError("INVALID_AUDIO", "参考片段采样率不一致", 400)
    rate = rates.pop() if rates else (sample_rate or SAMPLE_RATE)
    if sample_rate is not None and rate != sample_rate:
        raise AppError("INVALID_AUDIO", "参考片段采样率不一致", 400)
    if chunks:
        chunks.pop()
    merged = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    write_pcm16(dst, merged, rate)
    return len(merged) / rate


def select_reference(
    clips: list[tuple[Path, str]],
    dst_wav: Path,
    dst_txt: Path,
    min_sec: float = 15.0,
    max_sec: float = 30.0,
) -> float:
    """按顺序拼 2–4 个片段到 15–30 秒，并生成逐字转写。"""
    chosen: list[Path] = []
    texts: list[str] = []
    total = 0.0
    for path, text in clips:
        if total >= min_sec:
            break
        clip_duration = duration_sec(path)
        if chosen and total + clip_duration > max_sec:
            break
        if not chosen and clip_duration > max_sec:
            raise AppError(
                "INVALID_AUDIO",
                f"单个片段超过 {max_sec:.0f} 秒，请提供更短片段以保证音文一致",
                400,
            )
        chosen.append(path)
        texts.append(text.strip())
        total += clip_duration
    required = max(min_sec, config.MIN_REFERENCE_SEC)
    if total < required:
        raise AppError("INVALID_AUDIO", f"有效录音不足 {required:.0f} 秒，无法作为参考音色", 400)
    concat_with_gap(chosen, dst_wav, gap_sec=0.2)
    joined = "".join(
        text if text and text[-1] in "。！？；：，、.!?;:," else text + "，"
        for text in texts
    )
    dst_txt.write_text(joined.strip(), encoding="utf-8")
    return duration_sec(dst_wav)


def _pink_noise(length: int, sample_rate: int, rms: float, seed: int = 20261031) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(length)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(length, 1.0 / sample_rate)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(np.maximum(freqs[1:], 1.0))
    scale[0] = 0.0
    pink = np.fft.irfft(spectrum * scale, n=length)
    current = float(np.sqrt(np.mean(pink**2)))
    if current <= 1e-12:
        return np.zeros(length, dtype=np.float32)
    return (pink / current * rms).astype(np.float32)


def _limit_peak(data: np.ndarray, ceiling: float = 0.891) -> np.ndarray:
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > ceiling:
        data = data * (ceiling / peak)
    return data


def _add_noise(src: Path, dst: Path, noise_dbfs: float, seed: int = 20261031) -> None:
    data, sample_rate = read_wav(src)
    target_rms = 10 ** (noise_dbfs / 20.0)
    noise = _pink_noise(len(data), sample_rate, target_rms, seed=seed)
    mixed = _limit_peak(data + noise)
    write_pcm16(dst, mixed, sample_rate)


def postprocess(src: Path, dst: Path, mode: str = "clean") -> None:
    """先 loudnorm，再按模式做 EQ/底噪；默认 clean 复刻 warmup。"""
    if mode not in config.POSTPROCESS_MODES:
        raise AppError("INVALID_REQUEST", f"未知后处理模式：{mode}", 400)
    normalized = config.TMP_DIR / f"norm_{uuid.uuid4().hex}.wav"
    try:
        _run(
            [
                _ffmpeg(),
                "-y",
                "-i",
                str(src),
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                "-af",
                "loudnorm=I=-18:TP=-2:LRA=11",
                str(normalized),
            ]
        )
        if mode == "clean":
            shutil.copyfile(normalized, dst)
            return
        eq = (
            "highshelf=f=6000:g=-2.5,lowpass=f=16000"
            if mode == "natural"
            else "highshelf=f=6000:g=-4,equalizer=f=250:t=q:w=1:g=1.5,lowpass=f=16000"
        )
        shaped = config.TMP_DIR / f"eq_{uuid.uuid4().hex}.wav"
        try:
            _run(
                [
                    _ffmpeg(),
                    "-y",
                    "-i",
                    str(normalized),
                    "-ac",
                    "1",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-c:a",
                    "pcm_s16le",
                    "-af",
                    eq,
                    str(shaped),
                ]
            )
            _add_noise(shaped, dst, -55.0 if mode == "natural" else -52.0)
        finally:
            try:
                os.remove(shaped)
            except OSError:
                pass
    finally:
        try:
            os.remove(normalized)
        except OSError:
            pass


def peak_and_rms(path: Path) -> tuple[float, float]:
    data, _ = read_wav(path)
    if data.size == 0:
        return 0.0, -120.0
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(data**2)))
    peak_db = 20 * math.log10(max(peak, 1e-9))
    rms_db = 20 * math.log10(max(rms, 1e-9))
    return peak_db, rms_db
