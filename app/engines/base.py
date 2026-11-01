"""引擎协议与结果类型。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class SegmentResult:
    waveform: np.ndarray
    sample_rate: int
    infer_sec: float
    vram_peak_mb: float


class TTSEngine(Protocol):
    name: str
    sample_rate: int

    @property
    def is_ready(self) -> bool: ...

    @property
    def error(self) -> str | None: ...

    def load(self) -> None: ...

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
    ) -> SegmentResult: ...

    def clear_prompt_cache(self) -> None: ...

    def status(self) -> dict: ...
