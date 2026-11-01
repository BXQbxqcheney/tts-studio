"""混合引擎：GPT-SoVITS 双实例池 + Ming 回退，长任务自动并行。"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from app import config
from app.engines.base import SegmentResult, TTSEngine
from app.engines.gptsovits import GPTSoVITSEngine
from app.engines.ming import MingEngine

logger = logging.getLogger("tts_studio.hybrid")


class HybridEngine:
    name = "gptsovits"
    sample_rate = 32000

    def __init__(
        self,
        primary: TTSEngine | None = None,
        fallback: TTSEngine | None = None,
        fallback_below: int | None = None,
        secondary: TTSEngine | None = None,
        parallelism: int | None = None,
    ) -> None:
        self.primary = primary or GPTSoVITSEngine()
        if secondary is not None:
            self.secondary: TTSEngine | None = secondary
        elif config.GPTSOVITS_INSTANCES >= 2:
            self.secondary = GPTSoVITSEngine(port=config.GPTSOVITS_SECOND_PORT)
        else:
            self.secondary = None
        self.parallelism_limit = max(1, int(parallelism or config.GPTSOVITS_INSTANCES))
        self.fallback = fallback if fallback is not None else MingEngine()
        self.fallback_below = int(fallback_below or config.MING_FALLBACK_CHARS)
        self._cv = threading.Condition()
        self._available: list[TTSEngine] = [self.primary]
        self._job_lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self.primary.is_ready

    @property
    def error(self) -> str | None:
        return self.primary.error

    @property
    def parallelism(self) -> int:
        if self.secondary is not None and self.secondary.is_ready:
            return 2
        return 1

    def load(self) -> None:
        self.primary.load()

    def ensure_instances(self, count: int) -> int:
        count = 2 if count >= 2 and self.secondary is not None else 1
        if count >= 2:
            assert self.secondary is not None
            if not self.secondary.is_ready:
                self.secondary.load()
            with self._cv:
                if self.secondary not in self._available:
                    self._available.append(self.secondary)
                self._cv.notify_all()
            return 2
        if self.secondary is not None:
            with self._cv:
                if self.secondary in self._available:
                    self._available.remove(self.secondary)
                self._cv.notify_all()
            if self.secondary.is_ready:
                self.secondary.stop()
        return 1

    def begin_long_job(self, model: str | None = None, mode: str | None = None) -> int:
        choice = model or config.DEFAULT_MODEL
        preset = config.RESOURCE_MODES.get(mode or config.DEFAULT_MODE, config.RESOURCE_MODES["extreme"])
        with self._job_lock:
            if choice == "ming":
                self.ensure_instances(1)
                return 1
            unload = getattr(self.fallback, "unload", None)
            if self.fallback.is_ready and callable(unload):
                unload()
            try:
                return self.ensure_instances(int(preset.get("instances", 1)))
            except Exception:  # noqa: BLE001
                logger.exception("secondary GPT-SoVITS instance failed; running single instance")
                return 1

    def end_long_job(self) -> None:
        return

    def _acquire(self) -> TTSEngine:
        with self._cv:
            while not self._available:
                self._cv.wait(timeout=5.0)
            return self._available.pop()

    def _release(self, engine: TTSEngine) -> None:
        with self._cv:
            self._available.append(engine)
            self._cv.notify_all()

    def _fallback_ready(self) -> bool:
        if self.fallback.is_ready:
            return True
        try:
            self.fallback.load()
            return True
        except Exception:  # noqa: BLE001
            logger.exception("Ming fallback load failed; using GPT-SoVITS")
            return False

    def _use_fallback(
        self,
        text: str,
        voice_wav: Path,
        voice_text: str,
        task: str,
        controls: dict[str, str] | None,
        flow_steps: int,
    ) -> SegmentResult | None:
        self.ensure_instances(1)
        if not self._fallback_ready():
            return None
        return self.fallback.synthesize(text, voice_wav, voice_text, task, controls, flow_steps)

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
        choice = model or config.DEFAULT_MODEL
        if choice == "ming":
            result = self._use_fallback(text, voice_wav, voice_text, task, controls, flow_steps)
            if result is not None:
                logger.info("model=ming segment (%s chars)", len(text))
                return result
            choice = "gptsovits"
        elif choice == "gptsovits" and len(text) < self.fallback_below:
            result = self._use_fallback(text, voice_wav, voice_text, task, controls, flow_steps)
            if result is not None:
                logger.info("short segment (%s chars) routed to Ming fallback", len(text))
                return result
        engine = self._acquire()
        try:
            return engine.synthesize(text, voice_wav, voice_text, task, controls, flow_steps, model=choice, mode=mode)
        finally:
            self._release(engine)

    def clear_prompt_cache(self) -> None:
        self.primary.clear_prompt_cache()
        if self.secondary is not None and self.secondary.is_ready:
            self.secondary.clear_prompt_cache()
        if self.fallback.is_ready:
            self.fallback.clear_prompt_cache()

    def stop(self) -> None:
        for engine in (self.primary, self.secondary, self.fallback):
            if engine is None:
                continue
            close = getattr(engine, "stop", None)
            if callable(close):
                close()

    def status(self) -> dict:
        status = dict(self.primary.status())
        status["fallback_loaded"] = bool(self.fallback.is_ready)
        status["fallback_below"] = self.fallback_below
        status["secondary_loaded"] = bool(self.secondary is not None and self.secondary.is_ready)
        status["parallelism"] = self.parallelism
        status["models"] = list(config.MODEL_CHOICES)
        return status
