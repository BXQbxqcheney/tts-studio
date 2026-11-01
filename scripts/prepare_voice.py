"""从 data/raw 的逐句录音生成音色参考段、转写与音色注册。

用法：
  python scripts/prepare_voice.py --name "我的声音" --raw data/raw
  python scripts/prepare_voice.py --name "我的声音" --voice-id my_voice --min-sec 10 --max-sec 15
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.errors import AppError  # noqa: E402
from app.services import audio  # noqa: E402
from app.services.voices import VoiceRegistry  # noqa: E402

ALLOWED_EXT = (".wav", ".m4a", ".mp3", ".flac", ".ogg")


def read_transcript(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise AppError("INVALID_REQUEST", f"转写文件不存在：{path}", 400)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    mapping: dict[str, str] = {}
    for line in lines:
        line = line.strip()
        if not line or "|" not in line:
            continue
        stem, text = line.split("|", 1)
        stem = stem.strip()
        text = text.strip()
        if stem and text:
            mapping[stem] = text
    if not mapping:
        raise AppError("INVALID_REQUEST", "转写文件没有有效行（格式：001| 文本）", 400)
    return mapping


def find_audio(raw_dir: Path, stem: str) -> Path | None:
    for suffix in ALLOWED_EXT:
        candidate = raw_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    for candidate in raw_dir.glob(f"{stem}.*"):
        if candidate.suffix.lower() in ALLOWED_EXT and candidate.is_file():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="从逐句录音生成 Ming 参考音色")
    parser.add_argument("--raw", default=str(config.RAW_DIR), help="原始录音目录")
    parser.add_argument("--transcript", default="", help="转写文件，默认 <raw>/transcript.txt")
    parser.add_argument("--name", required=True, help="音色显示名")
    parser.add_argument("--voice-id", default="", help="可选 ASCII 音色 id")
    parser.add_argument("--description", default="", help="可选备注")
    parser.add_argument("--min-sec", type=float, default=4.0, help="参考段最短秒数（默认 4 秒）")
    parser.add_argument("--max-sec", type=float, default=8.0, help="参考段最长秒数（默认 8 秒）")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    transcript_path = Path(args.transcript) if args.transcript else raw_dir / "transcript.txt"
    voice_id = args.voice_id.strip() or ("v" + secrets.token_hex(4))
    try:
        transcript = read_transcript(transcript_path)
        clips: list[tuple[Path, str]] = []
        missing: list[str] = []
        processed_dir = config.PROCESSED_DIR / voice_id
        processed_dir.mkdir(parents=True, exist_ok=True)
        for stem in sorted(transcript):
            source = find_audio(raw_dir, stem)
            if source is None:
                missing.append(stem)
                continue
            destination = processed_dir / f"{stem}.wav"
            duration = audio.convert_reference(source, destination)
            clips.append((destination, transcript[stem]))
            print(f"ok {stem} {duration:.2f}s -> {destination}", flush=True)
        if missing:
            raise AppError("INVALID_AUDIO", f"缺少录音文件：{', '.join(missing)}", 400)
        if not clips:
            raise AppError("INVALID_AUDIO", "没有可用录音", 400)

        reference_wav = config.TMP_DIR / f"{voice_id}_reference.wav"
        reference_txt = config.TMP_DIR / f"{voice_id}_reference.txt"
        duration = audio.select_reference(
            clips,
            reference_wav,
            reference_txt,
            min_sec=args.min_sec,
            max_sec=args.max_sec,
        )
        registry = VoiceRegistry()
        voice = registry.create(
            name=args.name,
            upload_path=reference_wav,
            transcript=reference_txt.read_text(encoding="utf-8"),
            description=args.description,
            voice_id=voice_id,
        )
        summary = {
            "voice_id": voice.voice_id,
            "name": voice.name,
            "reference_duration_sec": round(duration, 3),
            "processed_clips": len(clips),
            "processed_dir": str(processed_dir),
            "voice_dir": str(config.VOICES_DIR / voice.voice_id),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        reference_wav.unlink(missing_ok=True)
        reference_txt.unlink(missing_ok=True)
        return 0
    except AppError as exc:
        print(f"ERROR {exc.code}: {exc.message}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
