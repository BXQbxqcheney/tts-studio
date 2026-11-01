"""通过镜像下载 Ming-3.0-tts 官方代码并解压，供适配层调用。"""

import json
import shutil
import sys
import traceback
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

TARGET_ROOT = Path(r"third_party")
ZIP_PATH = TARGET_ROOT / "Ming-3.0-tts-main.zip"
EXTRACT_ROOT = TARGET_ROOT / "Ming-3.0-tts-main"
SENTINEL = TARGET_ROOT / "CODE_READY.json"
LOG = Path(r".\logs\fetch_ming_code.log")

URLS = [
    "https://gh-proxy.com/https://github.com/inclusionAI/Ming-3.0-tts/archive/refs/heads/main.zip",
    "https://gh-proxy.com/https://codeload.github.com/inclusionAI/Ming-3.0-tts/zip/refs/heads/main",
    "https://gitclone.com/github.com/inclusionAI/Ming-3.0-tts/archive/refs/heads/main.zip",
]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def download(url: str) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response, ZIP_PATH.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)


def main() -> int:
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    try:
        for url in URLS:
            try:
                print(f"[{now()}] try {url}", flush=True)
                download(url)
                if ZIP_PATH.stat().st_size < 10_000:
                    raise ValueError(f"zip too small: {ZIP_PATH.stat().st_size}")
                break
            except Exception as exc:
                errors.append(f"{url}: {exc!r}")
                print(f"failed: {exc!r}", flush=True)
        else:
            raise RuntimeError("all mirrors failed: " + " | ".join(errors))
        with zipfile.ZipFile(ZIP_PATH) as archive:
            archive.extractall(TARGET_ROOT)
        candidates = [p for p in TARGET_ROOT.iterdir() if p.is_dir() and p.name.lower().startswith("ming-3.0-tts")]
        if EXTRACT_ROOT.exists():
            code_root = EXTRACT_ROOT
        elif candidates:
            code_root = candidates[0]
        else:
            raise RuntimeError("extracted directory not found")
        required = ["infer.py", "requirements.txt"]
        missing = [name for name in required if not (code_root / name).exists()]
        status = "ok" if not missing else "incomplete"
        SENTINEL.write_text(
            json.dumps(
                {
                    "stage": "ming_code",
                    "status": status,
                    "code_root": str(code_root),
                    "zip": str(ZIP_PATH),
                    "missing": missing,
                    "errors": errors,
                    "updated_at": now(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[{now()}] status={status} code_root={code_root} missing={missing}", flush=True)
        return 0 if status == "ok" else 2
    except Exception as exc:
        SENTINEL.write_text(
            json.dumps(
                {
                    "stage": "ming_code",
                    "status": "error",
                    "error": repr(exc),
                    "errors": errors,
                    "traceback": traceback.format_exc(),
                    "updated_at": now(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
