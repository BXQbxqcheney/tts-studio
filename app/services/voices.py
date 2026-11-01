"""音色注册表：上传、校验、元数据与删除。"""

from __future__ import annotations

import json
import logging
import re
import secrets
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from app import config
from app.errors import AppError
from app.schemas import VoiceInfo
from app.services import audio

logger = logging.getLogger("tts_studio.voices")
VOICE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,31}$")


class VoiceRegistry:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or config.VOICES_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def _voice_dir(self, voice_id: str) -> Path:
        if not VOICE_ID_RE.fullmatch(voice_id):
            raise AppError("VOICE_NOT_FOUND", "音色不存在", 404)
        return self.root / voice_id

    def _load_meta(self, voice_id: str) -> dict:
        path = self._voice_dir(voice_id) / "meta.json"
        if not path.is_file():
            raise AppError("VOICE_NOT_FOUND", "音色不存在", 404)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("failed to read voice meta %s: %s", path, exc)
            raise AppError("VOICE_NOT_FOUND", "音色元数据损坏", 404) from exc

    def _to_info(self, voice_id: str, meta: dict) -> VoiceInfo:
        return VoiceInfo(
            voice_id=voice_id,
            name=str(meta.get("name", voice_id)),
            duration_sec=float(meta.get("duration_sec", 0.0)),
            sample_rate=int(meta.get("sample_rate", 44100)),
            created_at=str(meta.get("created_at", "")),
            description=str(meta.get("description", "")),
            audio_url=f"/api/voices/{voice_id}/audio",
        )

    def list(self) -> list[VoiceInfo]:
        voices: list[VoiceInfo] = []
        for directory in sorted(self.root.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
            if not directory.is_dir():
                continue
            try:
                meta = self._load_meta(directory.name)
                voices.append(self._to_info(directory.name, meta))
            except AppError:
                logger.warning("skip invalid voice directory %s", directory)
        return voices

    def get(self, voice_id: str) -> VoiceInfo:
        return self._to_info(voice_id, self._load_meta(voice_id))

    def audio_path(self, voice_id: str) -> Path:
        path = self._voice_dir(voice_id) / "reference.wav"
        if not path.is_file():
            raise AppError("VOICE_NOT_FOUND", "音色参考音频不存在", 404)
        return path

    def transcript(self, voice_id: str) -> str:
        path = self._voice_dir(voice_id) / "reference.txt"
        if not path.is_file():
            raise AppError("VOICE_NOT_FOUND", "音色转写不存在", 404)
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise AppError("VOICE_NOT_FOUND", "音色转写为空", 404)
        return text

    def create(
        self,
        name: str,
        upload_path: Path,
        transcript: str,
        description: str = "",
        voice_id: str | None = None,
    ) -> VoiceInfo:
        name = name.strip()
        transcript = transcript.strip()
        if not 1 <= len(name) <= 40:
            raise AppError("INVALID_REQUEST", "音色名称需为 1–40 字符", 400)
        if not 1 <= len(transcript) <= 2000:
            raise AppError("INVALID_REQUEST", "参考音频转写需为 1–2000 字符", 400)
        if voice_id is not None:
            if not VOICE_ID_RE.fullmatch(voice_id):
                raise AppError("INVALID_REQUEST", "voice_id 格式不合法", 400)
            if (self.root / voice_id).exists():
                raise AppError("INVALID_REQUEST", "voice_id 已存在", 400)
        else:
            voice_id = "v" + secrets.token_hex(4)
            while (self.root / voice_id).exists():
                voice_id = "v" + secrets.token_hex(4)
        temp_raw = config.TMP_DIR / f"voice_{uuid.uuid4().hex}.wav"
        try:
            duration = audio.convert_reference(upload_path, temp_raw)
            if duration > config.MAX_REFERENCE_SEC:
                raise AppError(
                    "INVALID_AUDIO",
                    f"参考音频超过 {config.MAX_REFERENCE_SEC:.0f} 秒，请裁剪到 3–30 秒并保持转写一致",
                    400,
                )
            if duration < config.MIN_REFERENCE_SEC:
                raise AppError(
                    "INVALID_AUDIO",
                    f"参考音频需至少 {config.MIN_REFERENCE_SEC:.0f} 秒，当前 {duration:.1f} 秒",
                    400,
                )
            voice_dir = self.root / voice_id
            voice_dir.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(temp_raw, voice_dir / "reference.wav")
            (voice_dir / "reference.txt").write_text(transcript, encoding="utf-8")
            meta = {
                "voice_id": voice_id,
                "name": name,
                "description": description.strip()[:200],
                "duration_sec": round(duration, 3),
                "sample_rate": audio.SAMPLE_RATE,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            (voice_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return self._to_info(voice_id, meta)
        except Exception:
            shutil.rmtree(self.root / voice_id, ignore_errors=True)
            raise
        finally:
            temp_raw.unlink(missing_ok=True)

    def delete(self, voice_id: str) -> None:
        directory = self._voice_dir(voice_id)
        if not directory.is_dir():
            raise AppError("VOICE_NOT_FOUND", "音色不存在", 404)
        shutil.rmtree(directory)
