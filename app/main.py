"""FastAPI 入口：鉴权、队列、音色、合成、历史与静态 WebUI。"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app import config
from app.engines.base import TTSEngine
from app.engines.hybrid import HybridEngine
from app.engines.ming import MingEngine
from app.errors import AppError, make_error, register_error_handlers
from app.schemas import (
    CancelResponse,
    HealthResponse,
    HistoryItem,
    TTSRequest,
    TTSResponse,
    VoiceInfo,
    VoiceList,
)
from app.services.history import HistoryStore
from app.services.voices import VoiceRegistry
from app.worker import TTSWorker

logger = logging.getLogger("tts_studio.main")
DATE_RE = re.compile(r"^\d{8}$")
AUDIO_RE = re.compile(r"^tts_[A-Za-z0-9-]+\.wav$")
ALLOWED_EXT = {".wav", ".m4a", ".mp3", ".flac", ".ogg"}
_LOCAL_ORIGINS = {
    f"http://127.0.0.1:{config.PORT}",
    f"http://localhost:{config.PORT}",
    f"http://[::1]:{config.PORT}",
}


def _configure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(config.LOGS_DIR / "server.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise AppError("INVALID_AUDIO", f"不支持的音频格式：{suffix or '未知'}", 400)
    total = 0
    try:
        with destination.open("wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > config.MAX_UPLOAD_BYTES:
                    raise AppError("UPLOAD_TOO_LARGE", "上传文件超过 20MB", 413)
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


def _provided_token(request: Request) -> str:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.headers.get("x-tts-token", "").strip()


def _origin_allowed(origin: str) -> bool:
    if not origin or origin == "null":
        return not origin
    extra = {
        item.strip()
        for item in __import__("os").environ.get("TTS_ALLOWED_ORIGINS", "").split(",")
        if item.strip()
    }
    return origin in _LOCAL_ORIGINS or origin in extra


def _build_engine() -> TTSEngine:
    if config.ENGINE == "ming":
        return MingEngine()
    return HybridEngine()


def create_app(
    engine: TTSEngine | None = None,
    api_token: str | None = None,
    allowed_hosts: list[str] | None = None,
) -> FastAPI:
    config.ensure_directories()
    _configure_logging()
    token = config.API_TOKEN if api_token is None else (api_token.strip() or None)
    if config.require_token() and not token:
        raise RuntimeError("非本机监听必须设置 TTS_API_TOKEN")

    app = FastAPI(title="TTS Studio", version=config.get_version(), docs_url=None, redoc_url=None)
    app.state.token = token
    app.state.engine = engine or _build_engine()
    app.state.voices = VoiceRegistry()
    app.state.history = HistoryStore()
    app.state.worker = TTSWorker(app.state.engine, app.state.voices, app.state.history)
    app.state.load_thread = None

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=allowed_hosts or config.allowed_hosts(),
    )
    register_error_handlers(app)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        request.state.request_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
        path = request.url.path
        if request.method in {"POST", "PUT", "PATCH", "DELETE"} and path.startswith("/api/"):
            origin = request.headers.get("origin", "")
            if not _origin_allowed(origin):
                return JSONResponse(
                    status_code=403,
                    content=make_error("FORBIDDEN", "Origin 不被允许", request.state.request_id),
                )
        if token and path.startswith("/api/") and path != "/api/health":
            provided = _provided_token(request)
            if not provided or not secrets.compare_digest(provided, token):
                return JSONResponse(
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                    content=make_error("UNAUTHORIZED", "缺少或错误的访问令牌", request.state.request_id),
                )
        response = await call_next(request)
        response.headers["X-Request-Id"] = request.state.request_id
        return response

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        def load_engine() -> None:
            try:
                app.state.engine.load()
            except Exception:  # noqa: BLE001
                logger.exception("engine load failed; service degraded")

        app.state.load_thread = threading.Thread(target=load_engine, name="tts-engine-load", daemon=True)
        app.state.load_thread.start()
        app.state.worker.start()
        try:
            yield
        finally:
            app.state.worker.stop()
            close = getattr(app.state.engine, "stop", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    logger.exception("engine stop failed")

    app.router.lifespan_context = lifespan
    app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(config.STATIC_DIR / "index.html")

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        status = app.state.engine.status()
        if status.get("model_loaded"):
            state = "ok"
        elif status.get("error"):
            state = "degraded"
        else:
            state = "loading"
        return HealthResponse(
            status=state,
            version=config.get_version(),
            engine=app.state.engine.name,
            model_loaded=bool(status.get("model_loaded")),
            device=str(status.get("device", "")),
            gpu=str(status.get("gpu", "N/A")),
            sample_rate=int(status.get("sample_rate", 44100)),
            vram_allocated_mb=float(status.get("vram_allocated_mb", 0.0)),
            vram_peak_mb=float(status.get("vram_peak_mb", 0.0)),
            queue_depth=app.state.worker.queue_depth,
            queue_capacity=config.QUEUE_SIZE,
            concurrency=config.MAX_CONCURRENCY,
            parallelism=int(getattr(app.state.engine, "parallelism", 1)),
            mode=config.DEFAULT_MODE,
            batch_size=int(config.RESOURCE_MODES[config.DEFAULT_MODE]["batch"]),
            active_jobs=app.state.worker.active_jobs,
            error=status.get("error"),
        )

    @app.get("/api/engines")
    async def engines() -> dict:
        return {
            "engines": [{"name": app.state.engine.name, "available": app.state.engine.is_ready}],
            "modes": [
                {"id": key, **value}
                for key, value in config.RESOURCE_MODES.items()
            ],
            "models": [
                {
                    "id": model_id,
                    "label": config.MODEL_LABELS[model_id],
                    "max_chars": config.MODEL_SEGMENT_CHARS[model_id],
                    "available": app.state.engine.is_ready,
                }
                for model_id in config.MODEL_CHOICES
            ],
        }

    @app.get("/api/voices", response_model=VoiceList)
    async def list_voices() -> VoiceList:
        return VoiceList(voices=app.state.voices.list())

    @app.post("/api/voices", response_model=VoiceInfo, status_code=201)
    async def create_voice(
        name: str = Form(...),
        transcript: str = Form(...),
        description: str = Form(""),
        reference_audio: UploadFile = File(...),
    ) -> VoiceInfo:
        temp_path = config.TMP_DIR / f"upload_{uuid.uuid4().hex}"
        try:
            await _save_upload(reference_audio, temp_path)
            voice = app.state.voices.create(name, temp_path, transcript, description)
            app.state.engine.clear_prompt_cache()
            return voice
        finally:
            temp_path.unlink(missing_ok=True)

    @app.get("/api/voices/{voice_id}/audio")
    async def voice_audio(voice_id: str) -> FileResponse:
        path = app.state.voices.audio_path(voice_id)
        return FileResponse(path, media_type="audio/wav", filename="reference.wav")

    @app.delete("/api/voices/{voice_id}", status_code=204)
    async def delete_voice(voice_id: str) -> Response:
        app.state.voices.delete(voice_id)
        app.state.engine.clear_prompt_cache()
        return Response(status_code=204)

    @app.post("/api/tts", response_model=TTSResponse)
    async def synthesize(request: TTSRequest) -> TTSResponse:
        if not app.state.engine.is_ready:
            raise AppError("MODEL_NOT_READY", "模型尚未加载完成", 503, detail=app.state.engine.error)
        if len(request.text) > config.MAX_SYNC_TEXT_CHARS:
            raise AppError(
                "TEXT_TOO_LONG",
                f"同步接口最多 {config.MAX_SYNC_TEXT_CHARS} 字；长文本请用 /api/tts/jobs 或 /api/tts/default/long",
                413,
            )
        voice = app.state.voices.get(request.voice_id)
        job = app.state.worker.submit(request, voice)
        try:
            return await asyncio.wait_for(
                asyncio.wrap_future(job.future),
                timeout=config.JOB_TIMEOUT_SEC,
            )
        except TimeoutError as exc:
            app.state.worker.cancel(job.request_id)
            raise AppError("TIMEOUT", "生成超时，请缩短文本或稍后重试", 504) from exc

    @app.post("/api/tts/default")
    async def default_speak(request: TTSRequest) -> Response:
        """固定使用 default 音色，直接返回 WAV，方便其他程序调用。"""
        if not app.state.engine.is_ready:
            raise AppError("MODEL_NOT_READY", "模型尚未加载完成", 503, detail=app.state.engine.error)
        if len(request.text) > config.MAX_SYNC_TEXT_CHARS:
            raise AppError(
                "TEXT_TOO_LONG",
                f"直连接口最多 {config.MAX_SYNC_TEXT_CHARS} 字；长文本请用 /api/tts/default/long",
                413,
            )
        payload = request.model_copy(update={"voice_id": "default"})
        voice = app.state.voices.get("default")
        job = app.state.worker.submit(payload, voice)
        try:
            result = await asyncio.wait_for(
                asyncio.wrap_future(job.future),
                timeout=config.JOB_TIMEOUT_SEC,
            )
        except TimeoutError as exc:
            app.state.worker.cancel(job.request_id)
            raise AppError("TIMEOUT", "生成超时，请缩短文本或稍后重试", 504) from exc
        parts = result.audio_url.strip("/").split("/")
        if len(parts) < 2:
            raise AppError("AUDIO_NOT_FOUND", "生成音频路径异常", 404)
        audio_path = (config.OUTPUTS_DIR / parts[-2] / parts[-1]).resolve()
        if not audio_path.is_relative_to(config.OUTPUTS_DIR.resolve()) or not audio_path.is_file():
            raise AppError("AUDIO_NOT_FOUND", "生成音频不存在", 404)
        return Response(
            content=audio_path.read_bytes(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="default.wav"',
                "X-Request-Id": result.request_id,
                "X-Voice": "default",
                "X-Model": result.model,
                "X-Duration": f"{result.duration_sec:.3f}",
                "X-RTF": f"{result.rtf:.3f}",
            },
        )

    def _start_long(request: TTSRequest, voice_id: str | None = None) -> dict:
        if not app.state.engine.is_ready:
            raise AppError("MODEL_NOT_READY", "模型尚未加载完成", 503, detail=app.state.engine.error)
        if len(request.text) > config.MAX_TEXT_CHARS:
            raise AppError("TEXT_TOO_LONG", f"长任务最多 {config.MAX_TEXT_CHARS} 字", 413)
        if voice_id is not None:
            request = request.model_copy(update={"voice_id": voice_id})
        voice = app.state.voices.get(request.voice_id)
        long = app.state.worker.submit_long(request, voice)
        return {
            "job_id": long.job_id,
            "status": long.status,
            "segments_total": long.segments_total,
            "status_url": f"/api/tts/jobs/{long.job_id}",
            "audio_url": None,
        }

    @app.post("/api/tts/jobs", status_code=202)
    async def submit_long_job(request: TTSRequest) -> JSONResponse:
        return JSONResponse(status_code=202, content=_start_long(request))

    @app.post("/api/tts/default/long", status_code=202)
    async def submit_default_long_job(request: TTSRequest) -> JSONResponse:
        return JSONResponse(status_code=202, content=_start_long(request, "default"))

    @app.get("/api/tts/jobs/{job_id}")
    async def get_long_job(job_id: str) -> dict:
        status = app.state.worker.long_status(job_id)
        if status is None:
            raise AppError("JOB_NOT_FOUND", "长任务不存在", 404)
        return status

    @app.post("/api/tts/jobs/{job_id}/cancel")
    async def cancel_long_job(job_id: str) -> dict:
        if app.state.worker.long_status(job_id) is None:
            raise AppError("JOB_NOT_FOUND", "长任务不存在", 404)
        scope = app.state.worker.cancel_long(job_id)
        return {"job_id": job_id, "cancel_scope": scope}

    @app.post("/api/tts/jobs/{job_id}/resume", status_code=202)
    async def resume_long_job(job_id: str) -> JSONResponse:
        status = app.state.worker.resume_long(job_id)
        return JSONResponse(status_code=202, content=status)

    @app.post("/api/tts/{request_id}/cancel", response_model=CancelResponse)
    async def cancel(request_id: str) -> CancelResponse:
        scope = app.state.worker.cancel(request_id)
        return CancelResponse(request_id=request_id, cancel_scope=scope)

    @app.get("/api/history")
    async def history(limit: int = Query(5, ge=1, le=20)) -> dict:
        items: list[HistoryItem] = app.state.history.list(limit)
        return {"items": [item.model_dump() for item in items]}

    @app.get("/api/audio/{date}/{filename}")
    async def output_audio(date: str, filename: str) -> FileResponse:
        if not DATE_RE.fullmatch(date) or not AUDIO_RE.fullmatch(filename):
            raise AppError("AUDIO_NOT_FOUND", "音频不存在", 404)
        path = (config.OUTPUTS_DIR / date / filename).resolve()
        if not path.is_relative_to(config.OUTPUTS_DIR.resolve()) or not path.is_file():
            raise AppError("AUDIO_NOT_FOUND", "音频不存在", 404)
        return FileResponse(path, media_type="audio/wav", filename=filename)

    return app


app = create_app()
