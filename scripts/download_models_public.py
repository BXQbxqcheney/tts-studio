# -*- coding: utf-8 -*-
"""一键下载开源所需模型与代码（ModelScope / GitHub 镜像）。不包含任何用户权重。"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MING_ID = "inclusionAI/Ming-3.0-tts-0.5B"
ASR_ID = "pengzhendong/faster-whisper-small"
GPT_URL = (
    "https://www.modelscope.cn/models/FlowerCry/gpt-sovits-7z-pacakges/resolve/master/"
    "GPT-SoVITS-v2pro-20250604.7z"
)
MING_CODE_URLS = [
    "https://gh-proxy.com/https://github.com/inclusionAI/Ming-3.0-tts/archive/refs/heads/main.zip",
    "https://gh-proxy.com/https://codeload.github.com/inclusionAI/Ming-3.0-tts/zip/refs/heads/main",
    "https://gitclone.com/github.com/inclusionAI/Ming-3.0-tts/archive/refs/heads/main.zip",
]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)


def fetch_ming(dest: Path) -> None:
    from modelscope import snapshot_download

    target = dest / "Ming-3.0-tts-0.5B"
    print(f"[{now()}] Ming model -> {target}", flush=True)
    snapshot_download(MING_ID, local_dir=str(target))
    code_root = ROOT / "third_party"
    code_root.mkdir(parents=True, exist_ok=True)
    zip_path = code_root / "Ming-3.0-tts-main.zip"
    if not (code_root / "Ming-3.0-tts-main" / "ming_tts.py").exists():
        errors = []
        for url in MING_CODE_URLS:
            try:
                print(f"[{now()}] Ming code {url}", flush=True)
                download(url, zip_path)
                with zipfile.ZipFile(zip_path) as archive:
                    archive.extractall(code_root)
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{url}: {exc!r}")
        else:
            raise RuntimeError("Ming code download failed: " + " | ".join(errors))
    print(f"[{now()}] Ming code ready: {code_root / 'Ming-3.0-tts-main'}", flush=True)


def fetch_gpt(dest: Path) -> None:
    gpt_dir = dest / "GPT-SoVITS"
    package = gpt_dir / "GPT-SoVITS-v2pro-20250604"
    if (package / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt").exists():
        print(f"[{now()}] GPT-SoVITS already present: {package}", flush=True)
        return
    archive = gpt_dir / "GPT-SoVITS-v2pro-20250604.7z"
    gpt_dir.mkdir(parents=True, exist_ok=True)
    if not archive.exists() or archive.stat().st_size < 7_000_000_000:
        print(f"[{now()}] downloading GPT-SoVITS package (~7GB)", flush=True)
        download(GPT_URL, archive)
    print(f"[{now()}] extracting {archive}", flush=True)
    try:
        import py7zr

        with py7zr.SevenZipFile(archive, mode="r") as z:
            z.extractall(path=gpt_dir)
    except ImportError:
        raise RuntimeError("缺少 py7zr，请先运行 download_models.cmd 安装依赖")
    if not package.exists():
        candidates = [p for p in gpt_dir.iterdir() if p.is_dir() and p.name.startswith("GPT-SoVITS")]
        if candidates:
            package = candidates[0]
    print(f"[{now()}] GPT-SoVITS ready: {package}", flush=True)


def fetch_asr(dest: Path) -> None:
    from modelscope import snapshot_download

    target = dest / "asr" / "faster-whisper-small"
    print(f"[{now()}] ASR model -> {target}", flush=True)
    snapshot_download(ASR_ID, local_dir=str(target))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default=str(ROOT / "models"))
    parser.add_argument("--skip-ming", action="store_true")
    parser.add_argument("--skip-gpt", action="store_true")
    parser.add_argument("--with-asr", action="store_true")
    args = parser.parse_args()
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    errors = []
    for name, fn in (("ming", fetch_ming), ("gpt", fetch_gpt)):
        if (name == "ming" and args.skip_ming) or (name == "gpt" and args.skip_gpt):
            continue
        try:
            fn(dest)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc!r}")
            print(f"[{now()}] {name} failed: {exc!r}", flush=True)
    if args.with_asr:
        try:
            fetch_asr(dest)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"asr: {exc!r}")
    sentinel = {
        "status": "ok" if not errors else "partial",
        "dest": str(dest),
        "errors": errors,
        "finished_at": now(),
    }
    (dest / "MODELS_READY.json").write_text(
        json.dumps(sentinel, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(sentinel, ensure_ascii=False, indent=2), flush=True)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
