from __future__ import annotations

from pathlib import Path

from app.engines.hybrid import HybridEngine
from tests.conftest import FakeEngine


class RecordingEngine(FakeEngine):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.calls: list[str] = []

    def synthesize(self, text, voice_wav, voice_text, task, controls, flow_steps, model=None, mode=None):
        self.calls.append(text)
        return super().synthesize(text, voice_wav, voice_text, task, controls, flow_steps, model=model, mode=mode)


def test_hybrid_routes_short_text_to_fallback():
    primary = RecordingEngine("primary")
    fallback = RecordingEngine("fallback")
    engine = HybridEngine(primary=primary, fallback=fallback, fallback_below=20)
    voice = Path("reference.wav")
    engine.synthesize("你好。", voice, "你好。", "TTS", None, 10)
    assert primary.calls == []
    assert fallback.calls == ["你好。"]


def test_hybrid_routes_long_text_to_primary():
    primary = RecordingEngine("primary")
    fallback = RecordingEngine("fallback")
    engine = HybridEngine(primary=primary, fallback=fallback, fallback_below=20)
    text = "这是一段超过二十个字的正常段落，用来验证主引擎路由。"
    engine.synthesize(text, Path("reference.wav"), "转写", "TTS", None, 10)
    assert primary.calls == [text]
    assert fallback.calls == []


def test_hybrid_falls_back_to_primary_when_fallback_load_fails():
    class BrokenEngine(RecordingEngine):
        def __init__(self, name: str) -> None:
            super().__init__(name)
            self._ready = False

        def load(self) -> None:
            raise RuntimeError("fallback unavailable")

    primary = RecordingEngine("primary")
    engine = HybridEngine(primary=primary, fallback=BrokenEngine("fallback"), fallback_below=20)
    engine.synthesize("你好。", Path("reference.wav"), "你好。", "TTS", None, 10)
    assert primary.calls == ["你好。"]


def test_explicit_ming_model_routes_to_fallback():
    primary = RecordingEngine("primary")
    fallback = RecordingEngine("fallback")
    engine = HybridEngine(primary=primary, fallback=fallback, fallback_below=20)
    engine.synthesize("这是一段超过二十个字的文本，但用户明确选择了 Ming 模型。", Path("reference.wav"), "转写", "TTS", None, 10, model="ming")
    assert fallback.calls and not primary.calls


def test_ft_model_routes_to_primary_even_when_short():
    primary = RecordingEngine("primary")
    fallback = RecordingEngine("fallback")
    engine = HybridEngine(primary=primary, fallback=fallback, fallback_below=20)
    engine.synthesize("你好。", Path("reference.wav"), "你好。", "TTS", None, 10, model="gptsovits_ft")
    assert primary.calls == ["你好。"] and not fallback.calls
