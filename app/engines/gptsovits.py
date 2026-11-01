"""GPT-SoVITS 边车客户端：自动拉起本地推理进程并经 HTTP 调用。"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from app import config
from app.engines.base import SegmentResult
from app.errors import AppError

logger = logging.getLogger("tts_studio.gptsovits")

_SPEED_FACTOR = {
    "慢速": 0.85,
    "中速": 1.0,
    "正常": 1.0,
    "快速": 1.15,
    "slow": 0.85,
    "normal": 1.0,
    "fast": 1.15,
}


class GPTSoVITSEngine:
    name = "gptsovits"
    sample_rate = 32000

    def __init__(
        self,
        url: str | None = None,
        runtime: Path | None = None,
        sidecar: Path | None = None,
        gpt_weight: Path | None = None,
        sovits_weight: Path | None = None,
        base_gpt: Path | None = None,
        pkg: Path | None = None,
        start_timeout: int | None = None,
        port: int | None = None,
    ) -> None:
        self.port = int(port or config.GPTSOVITS_PORT)
        self.url = (url or f"http://{config.GPTSOVITS_HOST}:{self.port}").rstrip("/")
        self.runtime = Path(runtime or config.GPTSOVITS_RUNTIME)
        self.sidecar = Path(sidecar or config.GPTSOVITS_SIDECAR)
        self.gpt_weight = Path(gpt_weight or config.GPTSOVITS_GPT_WEIGHT)
        self.sovits_weight = Path(sovits_weight or config.GPTSOVITS_SOVITS_WEIGHT)
        self.base_gpt = Path(base_gpt or config.GPTSOVITS_BASE_GPT)
        self.pkg = Path(pkg or config.GPTSOVITS_PKG)
        self.start_timeout = int(start_timeout or config.GPTSOVITS_START_TIMEOUT)
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._log_handle = None
        self._ready = False
        self._loading = False
        self._error: str | None = None
        self._gpu = "N/A"
        self._vram_peak = 0.0

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> str | None:
        return self._error

    def _health(self, timeout: float = 2.0) -> dict | None:
        import requests

        try:
            response = requests.get(f"{self.url}/health", timeout=timeout)
            if response.status_code != 200:
                return None
            return response.json()
        except Exception:  # noqa: BLE001
            return None

    def _spawn(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if not self.runtime.is_file():
            raise AppError("MODEL_NOT_READY", f"GPT-SoVITS 运行时不存在：{self.runtime}", 503)
        if not self.sidecar.is_file():
            raise AppError("MODEL_NOT_READY", f"GPT-SoVITS 边车脚本不存在：{self.sidecar}", 503)
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = config.LOGS_DIR / (f"gptsovits_sidecar_{self.port}.log" if self.port != config.GPTSOVITS_PORT else "gptsovits_sidecar.log")
        self._log_handle = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PYTHONIOENCODING"] = "utf-8"
        env["GPTSOVITS_PKG"] = str(self.pkg)
        env["GPTSOVITS_GPT"] = str(self.gpt_weight)
        env["GPTSOVITS_BASE_GPT"] = str(self.base_gpt)
        env["GPTSOVITS_SOVITS"] = str(self.sovits_weight)
        env["GPTSOVITS_PORT"] = str(self.port)
        if config.GPTSOVITS_ASR_GATE:
            env.pop("GPTSOVITS_ASR", None)
        else:
            env["GPTSOVITS_ASR"] = "disabled"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc = subprocess.Popen(
            [str(self.runtime), "-s", str(self.sidecar), "--host", config.GPTSOVITS_HOST, "--port", str(self.port)],
            cwd=str(self.pkg),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=flags,
        )
        logger.info("GPT-SoVITS sidecar started pid=%s", self._proc.pid)

    def load(self) -> None:
        with self._lock:
            if self._ready:
                return
            self._loading = True
            try:
                health = self._health()
                if not (health and health.get("ready")):
                    self._spawn()
                    deadline = time.time() + self.start_timeout
                    health = None
                    while time.time() < deadline:
                        if self._proc is not None and self._proc.poll() is not None:
                            raise RuntimeError(
                                f"GPT-SoVITS 边车退出，退出码 {self._proc.returncode}，日志 {config.LOGS_DIR / 'gptsovits_sidecar.log'}"
                            )
                        health = self._health()
                        if health and health.get("ready"):
                            break
                        if health and health.get("error"):
                            raise RuntimeError(str(health["error"]))
                        time.sleep(1.0)
                if not health or not health.get("ready"):
                    raise RuntimeError(f"GPT-SoVITS 启动超时（{self.start_timeout}s）")
                self._gpu = str(health.get("gpu") or "N/A")
                self._vram_peak = float(health.get("vram_peak_mb") or 0.0)
                self._ready = True
                self._error = None
                logger.info("GPT-SoVITS ready gpu=%s vram=%.0fMB", self._gpu, self._vram_peak)
            except AppError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._ready = False
                self._error = f"{type(exc).__name__}: {exc}"
                logger.exception("GPT-SoVITS load failed")
                raise AppError("MODEL_NOT_READY", self._error, 503) from exc
            finally:
                self._loading = False

    def _post(
        self,
        text: str,
        voice_wav: Path,
        voice_text: str,
        controls: dict[str, str] | None,
        model: str | None,
        mode: str | None,
    ) -> tuple[bytes, dict]:
        import requests

        speed = 1.0
        if controls:
            speed = _SPEED_FACTOR.get(str(controls.get("speed") or ""), 1.0)
        preset = config.RESOURCE_MODES.get(mode or config.DEFAULT_MODE, {})
        batch = int(preset.get("batch", 1))
        chunk_chars = int(preset.get("chunk", 160))
        payload = {
            "text": text,
            "ref_wav": str(voice_wav),
            "prompt_text": voice_text,
            "prompt_lang": "zh",
            "text_lang": "zh",
            "speed_factor": speed,
            "model": "ft" if model == "gptsovits_ft" else "base",
            "batch_size": batch,
            "chunk_chars": chunk_chars,
        }
        started = time.perf_counter()
        response = requests.post(f"{self.url}/tts", json=payload, timeout=900)
        if response.status_code != 200:
            try:
                detail = response.json().get("error")
            except Exception:  # noqa: BLE001
                detail = response.text[:500]
            raise AppError("GENERATION_FAILED", f"GPT-SoVITS 推理失败：{detail}", 500)
        elapsed = time.perf_counter() - started
        headers = {
            "infer_sec": float(response.headers.get("X-Infer-Sec") or elapsed),
            "vram_peak_mb": float(response.headers.get("X-Vram-Mb") or self._vram_peak),
            "sample_rate": int(response.headers.get("X-Sample-Rate") or self.sample_rate),
        }
        return response.content, headers

    def synthesize(
        self,
        text: str,
        voice_wav: Path,
        voice_text: str,
        task: str,
        controls: dict[str, str] | None,
        flow_steps: int,
        model: str | None = None,
        mode: str | None = None,
    ) -> SegmentResult:
        if not voice_wav.is_file():
            raise AppError("VOICE_NOT_FOUND", "音色参考音频不存在", 404)
        if not voice_text.strip():
            raise AppError("VOICE_NOT_FOUND", "音色转写为空", 404)
        if not self._ready:
            self.load()
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                content, meta = self._post(text, voice_wav, voice_text, controls, model, mode)
                import soundfile as sf

                waveform, sample_rate = sf.read(io.BytesIO(content), dtype="float32", always_2d=False)
                if getattr(waveform, "ndim", 1) > 1:
                    waveform = waveform.mean(axis=1)
                waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
                if waveform.size == 0:
                    raise AppError("GENERATION_FAILED", "GPT-SoVITS 返回空音频", 500)
                return SegmentResult(
                    waveform=waveform,
                    sample_rate=int(sample_rate or meta["sample_rate"]),
                    infer_sec=float(meta["infer_sec"]),
                    vram_peak_mb=float(meta["vram_peak_mb"]),
                )
            except AppError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("GPT-SoVITS request attempt %s failed: %s", attempt + 1, exc)
                self._ready = False
                try:
                    self.load()
                except AppError as load_exc:
                    raise load_exc from exc
        raise AppError("GENERATION_FAILED", f"GPT-SoVITS 请求失败：{last_error}", 500)

    def clear_prompt_cache(self) -> None:
        return

    def stop(self) -> None:
        import requests

        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                requests.post(f"{self.url}/shutdown", timeout=2)
            except Exception:  # noqa: BLE001
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        self._proc = None
        self._ready = False

    def status(self) -> dict:
        health = self._health(timeout=1.0) if self._ready else None
        if health:
            self._gpu = str(health.get("gpu") or self._gpu)
            self._vram_peak = float(health.get("vram_peak_mb") or self._vram_peak)
        return {
            "model_loaded": self._ready,
            "loading": self._loading,
            "device": "cuda",
            "gpu": self._gpu,
            "sample_rate": self.sample_rate,
            "model": (health or {}).get("model") or "base",
            "vram_allocated_mb": float((health or {}).get("vram_allocated_mb") or 0.0),
            "vram_peak_mb": self._vram_peak,
            "error": self._error or (health or {}).get("error"),
        }
