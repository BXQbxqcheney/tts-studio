"""API 请求/响应模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from app import config


class TTSRequest(BaseModel):
    text: str = Field(min_length=1)
    voice_id: str = Field(default="default", pattern=r"^[a-z0-9][a-z0-9_-]{1,31}$")
    model: str = config.DEFAULT_MODEL
    mode: str = config.DEFAULT_MODE
    client_request_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9-]{8,64}$")
    quality: str = config.DEFAULT_QUALITY
    postprocess: str = config.DEFAULT_POSTPROCESS
    speed: str | None = None
    pitch: str | None = None
    volume: str | None = None
    emotion: str | None = None

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("text 不能为空")
        return text

    @model_validator(mode="after")
    def validate_controls(self) -> "TTSRequest":
        if self.model not in config.MODEL_CHOICES:
            raise PydanticCustomError("invalid_request", f"model 必须是 {list(config.MODEL_CHOICES)}")
        if self.mode not in config.RESOURCE_MODES:
            raise PydanticCustomError("invalid_request", f"mode 必须是 {list(config.RESOURCE_MODES)}")
        if self.quality not in config.QUALITY_FLOW_STEPS:
            raise PydanticCustomError("invalid_request", f"quality 必须是 {sorted(config.QUALITY_FLOW_STEPS)}")
        if self.postprocess not in config.POSTPROCESS_MODES:
            raise PydanticCustomError("invalid_request", f"postprocess 必须是 {list(config.POSTPROCESS_MODES)}")
        pairs = (
            ("speed", self.speed, config.SPEED_VALUES),
            ("pitch", self.pitch, config.PITCH_VALUES),
            ("volume", self.volume, config.VOLUME_VALUES),
            ("emotion", self.emotion, config.EMOTION_VALUES),
        )
        for name, value, allowed in pairs:
            if value is not None and value not in allowed:
                raise PydanticCustomError(
                    "invalid_control_combination",
                    f"{name} 必须是 {list(allowed)} 之一",
                )
        if self.emotion and any((self.speed, self.pitch, self.volume)):
            raise PydanticCustomError(
                "invalid_control_combination",
                "emotion 不能与 speed/pitch/volume 同时使用",
            )
        return self

    def task_and_controls(self) -> tuple[str, dict[str, str] | None]:
        if self.emotion:
            return "EMOTION", {"emotion": self.emotion}
        controls = {
            name: value
            for name, value in (("speed", self.speed), ("pitch", self.pitch), ("volume", self.volume))
            if value
        }
        if controls:
            return "BASIC", controls
        return "TTS", None


class VoiceInfo(BaseModel):
    voice_id: str
    name: str
    duration_sec: float
    sample_rate: int
    created_at: str
    description: str = ""
    audio_url: str


class VoiceList(BaseModel):
    voices: list[VoiceInfo]


class TTSResponse(BaseModel):
    request_id: str
    model: str = ""
    mode: str = ""
    audio_url: str
    duration_sec: float
    queue_sec: float
    infer_sec: float
    elapsed_sec: float
    rtf: float
    sample_rate: int
    flow_steps: int
    postprocess: str
    vram_peak_mb: float
    cancel_scope: str


class CancelResponse(BaseModel):
    request_id: str
    cancel_scope: Literal["queued", "between_segments", "not_cancellable", "completed"]


class HistoryItem(BaseModel):
    request_id: str
    voice_id: str
    text_preview: str
    audio_url: str
    duration_sec: float
    elapsed_sec: float
    rtf: float
    quality: str
    postprocess: str
    created_at: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "loading"]
    version: str
    engine: str
    model_loaded: bool
    device: str
    gpu: str
    sample_rate: int
    vram_allocated_mb: float
    vram_peak_mb: float
    queue_depth: int
    queue_capacity: int
    concurrency: int = 0
    parallelism: int = 1
    mode: str = ""
    batch_size: int = 1
    active_jobs: int = 0
    error: str | None = None


class ErrorInfo(BaseModel):
    code: str
    message: str
    request_id: str
    detail: Any | None = None
