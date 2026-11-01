"""运行配置与上游常量单源（开源脱敏版：模型/数据/运行时均相对仓库目录）。"""

from __future__ import annotations

import ipaddress
import os
import sys
import tomllib
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
VOICES_DIR = DATA_DIR / "voices"
PROCESSED_DIR = DATA_DIR / "processed"
TMP_DIR = DATA_DIR / "tmp"
OUTPUTS_DIR = ROOT / "outputs"
JOBS_DIR = OUTPUTS_DIR / "jobs"
LOGS_DIR = ROOT / "logs"
STATIC_DIR = ROOT / "app" / "static"
HISTORY_FILE = OUTPUTS_DIR / "history.jsonl"

MODEL_DIR = Path(os.environ.get("TTS_MODEL_DIR", str(ROOT / "models" / "Ming-3.0-tts-0.5B")))
CODE_DIR = Path(os.environ.get("TTS_CODE_DIR", str(ROOT / "third_party" / "Ming-3.0-tts-main")))

ENGINE = os.environ.get("TTS_ENGINE", "gptsovits").strip().lower()
GPTSOVITS_PKG = Path(
    os.environ.get(
        "TTS_GPTSOVITS_PKG",
        str(ROOT / "models" / "GPT-SoVITS" / "GPT-SoVITS-v2pro-20250604"),
    )
)
GPTSOVITS_GPT_WEIGHT = Path(
    os.environ.get(
        "TTS_GPTSOVITS_GPT",
        str(GPTSOVITS_PKG / "GPT_weights_v2ProPlus" / "custom.ckpt"),
    )
)
GPTSOVITS_SOVITS_WEIGHT = Path(
    os.environ.get(
        "TTS_GPTSOVITS_SOVITS",
        str(GPTSOVITS_PKG / "GPT_SoVITS" / "pretrained_models" / "v2Pro" / "s2Gv2ProPlus.pth"),
    )
)
GPTSOVITS_BASE_GPT = Path(
    os.environ.get(
        "TTS_GPTSOVITS_BASE_GPT",
        str(GPTSOVITS_PKG / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt"),
    )
)
GPTSOVITS_RUNTIME = Path(os.environ.get("TTS_GPTSOVITS_RUNTIME", str(ROOT / "runtime" / "python.exe")))
GPTSOVITS_SIDECAR = ROOT / "scripts" / "gptsovits_sidecar.py"
GPTSOVITS_HOST = os.environ.get("TTS_GPTSOVITS_HOST", "127.0.0.1").strip()
GPTSOVITS_PORT = int(os.environ.get("TTS_GPTSOVITS_PORT", "8766"))
GPTSOVITS_SECOND_PORT = int(os.environ.get("TTS_GPTSOVITS_SECOND_PORT", str(GPTSOVITS_PORT + 1)))
GPTSOVITS_INSTANCES = max(1, min(2, int(os.environ.get("TTS_GPTSOVITS_INSTANCES", "2"))))
GPTSOVITS_ASR_GATE = os.environ.get("TTS_GPTSOVITS_ASR_GATE", "0").strip() == "1"
GPTSOVITS_URL = f"http://{GPTSOVITS_HOST}:{GPTSOVITS_PORT}"
GPTSOVITS_SECOND_URL = f"http://{GPTSOVITS_HOST}:{GPTSOVITS_SECOND_PORT}"
GPTSOVITS_START_TIMEOUT = int(os.environ.get("TTS_GPTSOVITS_START_TIMEOUT", "180"))
MING_FALLBACK_CHARS = int(os.environ.get("TTS_MING_FALLBACK_CHARS", "0"))

HOST = os.environ.get("TTS_HOST", "127.0.0.1").strip()
PORT = int(os.environ.get("TTS_PORT", "8765"))
API_TOKEN = os.environ.get("TTS_API_TOKEN", "").strip() or None
DEVICE = os.environ.get("TTS_DEVICE", "cuda:0")


def _default_concurrency() -> int:
    if ENGINE == "gptsovits":
        return 1
    try:
        import psutil

        cores = psutil.cpu_count(logical=False) or (os.cpu_count() or 4)
    except Exception:
        cores = os.cpu_count() or 4
    return max(1, int(cores) // 2)


MAX_CONCURRENCY = max(1, int(os.environ.get("TTS_MAX_CONCURRENCY", _default_concurrency())))
QUEUE_SIZE = max(16, MAX_CONCURRENCY * 2)
QUEUE_TIMEOUT_SEC = 300
JOB_TIMEOUT_SEC = 600
MAX_TEXT_CHARS = 10_000
MAX_SYNC_TEXT_CHARS = 500
MAX_SEGMENT_CHARS = 80
MIN_SEGMENT_CHARS = int(os.environ.get("TTS_MIN_SEGMENT_CHARS", str(MING_FALLBACK_CHARS or 20)))
LONG_JOB_RETRIES = 2
LONG_JOB_GAP_SEC = 0.15
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_REFERENCE_SEC = 30.0
MIN_REFERENCE_SEC = 3.0
SEGMENT_SILENCE_SEC = 0.15

QUALITY_FLOW_STEPS = {"fast": 4, "balanced": 6, "quality": 10}
DEFAULT_QUALITY = "quality"
POSTPROCESS_MODES = ("clean", "natural", "warm")
DEFAULT_POSTPROCESS = "clean"

MODEL_CHOICES = ("gptsovits", "gptsovits_ft", "ming")
MODEL_LABELS = {
    "gptsovits": "GPT-SoVITS 长段（底模 GPT + SoVITS）",
    "gptsovits_ft": "GPT-SoVITS 微调（自备微调权重）",
    "ming": "Ming 零样本",
}
MODEL_SEGMENT_CHARS = {"gptsovits": 1000, "gptsovits_ft": 160, "ming": 80}
RESOURCE_MODES = {
    "extreme": {"label": "极限", "instances": 1, "batch": 4},
    "high": {"label": "高", "instances": 2, "batch": 1},
    "higher": {"label": "较高", "instances": 1, "batch": 2},
    "medium": {"label": "中", "instances": 1, "batch": 1},
    "low": {"label": "低", "instances": 1, "batch": 1, "chunk": 120},
    "minimal": {"label": "极低", "instances": 1, "batch": 1, "chunk": 80},
}
DEFAULT_MODE = os.environ.get("TTS_RESOURCE_MODE", "extreme").strip().lower()
if DEFAULT_MODE not in RESOURCE_MODES:
    DEFAULT_MODE = "extreme"
DEFAULT_MODEL = os.environ.get("TTS_DEFAULT_MODEL", "ming" if ENGINE == "ming" else "gptsovits").strip().lower()
if DEFAULT_MODEL not in MODEL_CHOICES:
    DEFAULT_MODEL = "gptsovits"


class _MingDefaults:
    BASIC_ATTRIBUTE_VALUES = {
        "speed": ("慢速", "中速", "快速"),
        "pitch": ("低", "中", "高"),
        "volume": ("低", "中", "高"),
    }
    EMOTION_VALUES = ("高兴", "悲伤", "愤怒", "恐惧", "厌恶", "惊讶", "中性")


def _load_ming_tasks():
    code = str(CODE_DIR)
    if code not in sys.path:
        sys.path.insert(0, code)
    try:
        import tts_tasks  # noqa: PLC0415
    except Exception:
        return _MingDefaults()
    return tts_tasks


MING_TASKS = _load_ming_tasks()
SPEED_VALUES = tuple(MING_TASKS.BASIC_ATTRIBUTE_VALUES["speed"])
PITCH_VALUES = tuple(MING_TASKS.BASIC_ATTRIBUTE_VALUES["pitch"])
VOLUME_VALUES = tuple(MING_TASKS.BASIC_ATTRIBUTE_VALUES["volume"])
EMOTION_VALUES = tuple(MING_TASKS.EMOTION_VALUES)


def get_version() -> str:
    try:
        with (ROOT / "pyproject.toml").open("rb") as handle:
            return str(tomllib.load(handle)["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        try:
            return metadata.version("tts-studio")
        except metadata.PackageNotFoundError:
            return "0.0.0"


def is_loopback(host: str) -> bool:
    value = host.strip().strip("[]")
    if value.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def allowed_hosts() -> list[str]:
    values = {"127.0.0.1", "localhost", "[::1]", HOST}
    extra = os.environ.get("TTS_ALLOWED_HOSTS", "").strip()
    if extra:
        values.update(item.strip() for item in extra.split(",") if item.strip())
    return sorted(values)


def require_token() -> bool:
    return not is_loopback(HOST)


def ensure_directories() -> None:
    for path in (DATA_DIR, RAW_DIR, VOICES_DIR, PROCESSED_DIR, TMP_DIR, OUTPUTS_DIR, JOBS_DIR, LOGS_DIR, STATIC_DIR):
        path.mkdir(parents=True, exist_ok=True)


def validate_startup() -> None:
    if require_token() and not API_TOKEN:
        raise RuntimeError("非本机监听必须设置 TTS_API_TOKEN；拒绝在无鉴权情况下暴露服务。")
    if ENGINE == "ming":
        if not MODEL_DIR.is_dir():
            raise RuntimeError(f"Ming 模型目录不存在：{MODEL_DIR}")
        if not (CODE_DIR / "ming_tts.py").is_file():
            raise RuntimeError(f"Ming 官方代码目录不完整：{CODE_DIR}")
        return
    if not (GPTSOVITS_PKG / "GPT_SoVITS").is_dir():
        raise RuntimeError(f"GPT-SoVITS 目录不存在：{GPTSOVITS_PKG}")
    if not GPTSOVITS_BASE_GPT.is_file():
        raise RuntimeError(f"GPT-SoVITS 底模 GPT 不存在：{GPTSOVITS_BASE_GPT}")
    if not GPTSOVITS_SOVITS_WEIGHT.is_file():
        raise RuntimeError(f"GPT-SoVITS SoVITS 权重不存在：{GPTSOVITS_SOVITS_WEIGHT}")
    if not GPTSOVITS_RUNTIME.is_file():
        raise RuntimeError(f"GPT-SoVITS 运行时不存在：{GPTSOVITS_RUNTIME}")
    if not GPTSOVITS_SIDECAR.is_file():
        raise RuntimeError(f"GPT-SoVITS 边车脚本不存在：{GPTSOVITS_SIDECAR}")
