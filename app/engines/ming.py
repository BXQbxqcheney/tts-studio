"""Ming-3.0-tts 适配：延迟加载、flow steps、提示缓存与显存指标。"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

from app import config
from app.engines.base import SegmentResult
from app.errors import AppError

logger = logging.getLogger("tts_studio.ming")


class MingEngine:
    name = "ming"
    sample_rate = 44100

    def __init__(self, model_dir: Path | None = None, code_dir: Path | None = None, device: str | None = None) -> None:
        self.model_dir = Path(model_dir or config.MODEL_DIR)
        self.code_dir = Path(code_dir or config.CODE_DIR)
        self.device = device or config.DEVICE
        self._lock = threading.Lock()
        self._tls = threading.local()
        self._model = None
        self._torch = None
        self._flowloss = None
        self._original_sample = None
        self._ready = False
        self._loading = False
        self._error: str | None = None
        self._ivr_peak = 0.0

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> str | None:
        return self._error

    def load(self) -> None:
        if self._ready:
            return
        self._loading = True
        started = time.perf_counter()
        try:
            from loguru import logger as loguru_logger

            loguru_logger.remove()
            loguru_logger.add(
                str(config.LOGS_DIR / "ming_upstream.log"),
                level="ERROR",
                encoding="utf-8",
                rotation="10 MB",
                filter=lambda record: record["name"].startswith("modeling_bailingmm"),
            )

            code = str(self.code_dir)
            if code not in sys.path:
                sys.path.insert(0, code)
            import torch  # noqa: PLC0415
            import fm.flowloss as flowloss  # noqa: PLC0415
            from ming_tts import MingTTS  # noqa: PLC0415

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if self._original_sample is None:
                self._original_sample = flowloss.FlowLoss.sample
                original = self._original_sample
                tls = self._tls

                def sample(flow_self, z, latent_history, cfg=1.0, patch_size=1, sigma=0.25, temperature=0, **kwargs):
                    return original(
                        flow_self,
                        z,
                        latent_history,
                        cfg,
                        patch_size,
                        sigma,
                        temperature,
                        steps=getattr(tls, "steps", 10),
                    )

                flowloss.FlowLoss.sample = sample
            self._flowloss = flowloss
            self._torch = torch
            self._model = MingTTS(str(self.model_dir), device=self.device, text_normalization=False)
            self.sample_rate = int(self._model.sample_rate)
            self._ready = True
            self._error = None
            logger.info("Ming loaded in %.1fs on %s", time.perf_counter() - started, self.device)
        except Exception as exc:  # noqa: BLE001
            self._error = f"{type(exc).__name__}: {exc}"
            logger.exception("Ming load failed")
            raise
        finally:
            self._loading = False

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
        if not self._ready or self._model is None or self._torch is None:
            raise AppError("MODEL_NOT_READY", "模型尚未加载完成", 503, detail=self._error)
        if not voice_wav.is_file():
            raise AppError("VOICE_NOT_FOUND", "音色参考音频不存在", 404)
        if not voice_text.strip():
            raise AppError("VOICE_NOT_FOUND", "音色转写为空", 404)
        torch = self._torch
        self._tls.steps = max(1, min(int(flow_steps), 20))
        started = time.perf_counter()
        try:
            waveform = self._model.synthesize(
                task=task,
                language_id="Chinese",
                text=text,
                prompt_wav_path=str(voice_wav),
                prompt_text=voice_text,
                use_spk_emb=True,
                controls=controls,
                cfg=2.0,
                sigma=0.25,
                temperature=0.0,
                max_decode_steps=1200,
                trim_silence=True,
                raw_text=False,
            )
            array = waveform.detach().cpu().float().numpy().reshape(-1)
            infer_sec = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated() / 1024**2
            self._ivr_peak = float(peak)
            return SegmentResult(
                waveform=np.asarray(array, dtype=np.float32),
                sample_rate=self.sample_rate,
                infer_sec=infer_sec,
                vram_peak_mb=float(peak),
            )
        except AppError:
            raise
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            logger.exception("Ming inference failed")
            if "out of memory" in message.lower():
                torch.cuda.empty_cache()
                raise AppError("OOM", "显存不足，请关闭占显存程序后重试", 507) from exc
            raise AppError("GENERATION_FAILED", "语音生成失败，请查看日志", 500) from exc

    def clear_prompt_cache(self) -> None:
        with self._lock:
            if self._model is not None and hasattr(self._model, "_prompt_cache"):
                self._model._prompt_cache.clear()
                logger.info("Ming prompt cache cleared")

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._ready = False
            self._error = None
            if self._torch is not None and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
            import gc

            gc.collect()
            logger.info("Ming unloaded to free VRAM")

    def status(self) -> dict:
        allocated = 0.0
        gpu = "N/A"
        if self._torch is not None and self._torch.cuda.is_available():
            allocated = self._torch.cuda.memory_allocated() / 1024**2
            gpu = self._torch.cuda.get_device_name(0)
        return {
            "model_loaded": self._ready,
            "loading": self._loading,
            "device": self.device,
            "gpu": gpu,
            "sample_rate": self.sample_rate,
            "vram_allocated_mb": round(float(allocated), 1),
            "vram_peak_mb": round(float(self._ivr_peak), 1),
            "error": self._error,
        }
