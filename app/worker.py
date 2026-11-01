"""有界队列 + 单 worker；短请求同步返回，长文本后台任务可轮询/恢复。"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import queue
import secrets
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from app import config
from app.engines.base import TTSEngine
from app.errors import AppError
from app.schemas import HistoryItem, TTSRequest, TTSResponse, VoiceInfo
from app.services import audio
from app.services.history import HistoryStore
from app.services.text import segment_text
from app.services.voices import VoiceRegistry

logger = logging.getLogger("tts_studio.worker")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _segments_for(request: TTSRequest, engine_name: str | None = None) -> list[str]:
    if engine_name == "ming":
        max_chars = config.MODEL_SEGMENT_CHARS["ming"]
    else:
        max_chars = config.MODEL_SEGMENT_CHARS.get(request.model, config.MAX_SEGMENT_CHARS)
    return segment_text(request.text, max_chars=max_chars, min_chars=config.MIN_SEGMENT_CHARS)


def _status_from_data(data: dict) -> dict:
    now = datetime.now().astimezone()
    created = _parse_time(data.get("created_at"))
    started = _parse_time(data.get("started_at"))
    finished = _parse_time(data.get("finished_at"))
    total = int(data.get("segments_total") or 0)
    done = int(data.get("segments_done") or 0)
    chars_total = int(data.get("chars_total") or 0)
    chars_done = int(data.get("chars_done") or 0)
    progress = (
        round(chars_done / chars_total, 4)
        if chars_total
        else round(done / total, 4)
        if total
        else 0.0
    )
    queue_sec = 0.0
    if started and created:
        queue_sec = max(0.0, (started - created).total_seconds())
    elapsed = 0.0
    if started:
        elapsed = max(0.0, ((finished or now) - started).total_seconds())
    eta = None
    if started and chars_done > 0 and chars_total > chars_done:
        eta = round(elapsed / chars_done * (chars_total - chars_done), 1)
    elif started and done > 0 and done < total:
        eta = round(elapsed / done * (total - done), 1)
    status = str(data.get("status") or "unknown")
    return {
        "job_id": data.get("job_id", ""),
        "request_id": data.get("job_id", ""),
        "status": status,
        "segments_total": total,
        "segments_done": done,
        "chars_total": chars_total,
        "chars_done": chars_done,
        "current_segment": int(data.get("current_segment") or 0),
        "progress": progress,
        "created_at": data.get("created_at"),
        "started_at": data.get("started_at"),
        "finished_at": data.get("finished_at"),
        "elapsed_sec": round(elapsed, 1),
        "queue_sec": round(queue_sec, 1),
        "eta_sec": eta,
        "audio_url": data.get("audio_url"),
        "error": data.get("error"),
        "text_preview": data.get("text_preview", ""),
        "voice_id": data.get("voice_id", ""),
        "model": data.get("model", config.DEFAULT_MODEL),
        "mode": data.get("mode", config.DEFAULT_MODE),
        "quality": data.get("quality", ""),
        "duration_sec": float(data.get("duration_sec") or 0.0),
        "infer_sec": float(data.get("infer_sec") or 0.0),
        "rtf": float(data.get("rtf") or 0.0),
        "sample_rate": audio.SAMPLE_RATE,
        "flow_steps": int(data.get("flow_steps") or 0),
        "postprocess": data.get("postprocess", ""),
        "vram_peak_mb": float(data.get("vram_peak_mb") or 0.0),
        "cancel_scope": (
            "completed"
            if status == "done"
            else "between_segments"
            if status == "running"
            else "queued"
            if status == "queued"
            else "not_cancellable"
        ),
    }


@dataclass
class LongJob:
    job_id: str
    request: TTSRequest
    voice: VoiceInfo
    job_dir: Path
    created_at: str
    status: str = "queued"
    segments_total: int = 0
    segments_done: int = 0
    chars_total: int = 0
    chars_done: int = 0
    current_segment: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    audio_url: str | None = None
    infer_sec: float = 0.0
    vram_peak_mb: float = 0.0
    duration_sec: float = 0.0
    rtf: float = 0.0
    flow_steps: int = 0
    cancel_event: threading.Event = field(default_factory=threading.Event)
    progress_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def text_preview(self) -> str:
        return self.request.text[:40]

    def manifest(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "segments_total": self.segments_total,
            "segments_done": self.segments_done,
            "chars_total": self.chars_total,
            "chars_done": self.chars_done,
            "current_segment": self.current_segment,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "audio_url": self.audio_url,
            "infer_sec": round(self.infer_sec, 3),
            "vram_peak_mb": round(self.vram_peak_mb, 1),
            "duration_sec": round(self.duration_sec, 3),
            "rtf": round(self.rtf, 3),
            "flow_steps": self.flow_steps,
            "postprocess": self.request.postprocess,
            "text": self.request.text,
            "text_preview": self.text_preview,
            "voice_id": self.request.voice_id,
            "model": self.request.model,
            "mode": self.request.mode,
            "quality": self.request.quality,
            "speed": self.request.speed,
            "pitch": self.request.pitch,
            "volume": self.request.volume,
            "emotion": self.request.emotion,
            "client_request_id": self.request.client_request_id,
        }

    def status_dict(self) -> dict:
        return _status_from_data(self.manifest())


@dataclass
class Job:
    request_id: str
    request: TTSRequest
    voice: VoiceInfo
    created_at: float
    cancel_event: threading.Event = field(default_factory=threading.Event)
    future: concurrent.futures.Future = field(default_factory=concurrent.futures.Future)
    started_at: float | None = None
    status: str = "queued"
    long: LongJob | None = None


class TTSWorker:
    def __init__(
        self,
        engine: TTSEngine,
        voices: VoiceRegistry,
        history: HistoryStore,
        outputs_dir: Path | None = None,
    ) -> None:
        self.engine = engine
        self.voices = voices
        self.history = history
        self.outputs_dir = outputs_dir or config.OUTPUTS_DIR
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_root = config.JOBS_DIR
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[Job | None] = queue.Queue(maxsize=config.QUEUE_SIZE)
        self._jobs: dict[str, Job] = {}
        self.long_jobs: dict[str, LongJob] = {}
        self.long_manifests: dict[str, dict] = {}
        self._jobs_lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._stopping = threading.Event()

    def start(self) -> None:
        if any(thread.is_alive() for thread in self._threads):
            return
        self.load_long_jobs()
        self._stopping.clear()
        self._threads = [
            threading.Thread(target=self._loop, name=f"tts-worker-{index + 1}", daemon=True)
            for index in range(config.MAX_CONCURRENCY)
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stopping.set()
        with self._jobs_lock:
            for long in list(self.long_jobs.values()):
                if long.status in {"queued", "running", "cancelling"}:
                    long.cancel_event.set()
                    long.status = "interrupted"
                    long.error = "服务停止，任务可恢复"
                    long.finished_at = _now()
                    self._write_manifest(long)
                    self.long_manifests[long.job_id] = long.manifest()
        for _ in range(max(1, len(self._threads))):
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                try:
                    self._queue.put(None, timeout=1)
                except queue.Full:
                    break
        for thread in self._threads:
            thread.join(timeout=5)
        with self._jobs_lock:
            jobs = list(self._jobs.values())
            self._jobs.clear()
        for job in jobs:
            if not job.future.done():
                job.future.set_exception(AppError("SERVICE_STOPPED", "服务已停止", 503))

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def active_jobs(self) -> int:
        with self._jobs_lock:
            short_running = sum(1 for job in self._jobs.values() if job.status == "running")
        long_running = sum(
            1 for long in self.long_jobs.values() if long.status in {"running", "cancelling"}
        )
        return short_running + long_running

    def submit(self, request: TTSRequest, voice: VoiceInfo) -> Job:
        request_id = request.client_request_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
        with self._jobs_lock:
            if request_id in self._jobs:
                request_id = f"{request_id}-{secrets.token_hex(2)}"
        job = Job(
            request_id=request_id,
            request=request,
            voice=voice,
            created_at=time.perf_counter(),
        )
        with self._jobs_lock:
            self._jobs[request_id] = job
        try:
            self._queue.put_nowait(job)
        except queue.Full as exc:
            with self._jobs_lock:
                self._jobs.pop(request_id, None)
            raise AppError("QUEUE_FULL", "生成队列已满，请稍后重试", 429) from exc
        return job

    def submit_long(self, request: TTSRequest, voice: VoiceInfo) -> LongJob:
        job_id = request.client_request_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
        with self._jobs_lock:
            if job_id in self.long_jobs or job_id in self.long_manifests:
                job_id = f"{job_id}-{secrets.token_hex(2)}"
        segments = _segments_for(request, self.engine.name)
        job_dir = self.jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        long = LongJob(
            job_id=job_id,
            request=request,
            voice=voice,
            job_dir=job_dir,
            created_at=_now(),
            segments_total=len(segments),
            chars_total=sum(len(segment) for segment in segments),
        )
        self.long_jobs[job_id] = long
        self._write_manifest(long)
        self._enqueue_long(long)
        return long

    def _enqueue_long(self, long: LongJob) -> None:
        job = Job(
            request_id=long.job_id,
            request=long.request,
            voice=long.voice,
            created_at=time.perf_counter(),
            long=long,
        )
        try:
            self._queue.put_nowait(job)
        except queue.Full as exc:
            long.status = "error"
            long.error = "生成队列已满"
            long.finished_at = _now()
            self._write_manifest(long)
            raise AppError("QUEUE_FULL", "生成队列已满，请稍后重试", 429) from exc

    def long_status(self, job_id: str) -> dict | None:
        long = self.long_jobs.get(job_id)
        if long is not None:
            return long.status_dict()
        data = self.long_manifests.get(job_id)
        if data is None:
            path = self.jobs_root / job_id / "job.json"
            if path.is_file():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    self.long_manifests[job_id] = data
                except (OSError, json.JSONDecodeError):
                    logger.warning("failed to read job manifest %s", path)
                    return None
        return _status_from_data(data) if data else None

    def cancel_long(self, job_id: str) -> str:
        long = self.long_jobs.get(job_id)
        if long is None:
            return "not_cancellable"
        if long.status == "queued":
            long.cancel_event.set()
            long.status = "cancelled"
            long.finished_at = _now()
            self._write_manifest(long)
            return "queued"
        if long.status in {"running", "cancelling"}:
            long.cancel_event.set()
            long.status = "cancelling"
            self._write_manifest(long)
            return "between_segments"
        return "not_cancellable"

    def resume_long(self, job_id: str) -> dict:
        data = self.long_status(job_id)
        if data is None:
            raise AppError("JOB_NOT_FOUND", "长任务不存在", 404)
        if data["status"] in {"queued", "running", "cancelling"}:
            raise AppError("JOB_BUSY", "任务正在运行", 409)
        if data["status"] == "done":
            raise AppError("JOB_BUSY", "任务已完成", 409)
        manifest = self.long_manifests.get(job_id)
        if manifest is None:
            path = self.jobs_root / job_id / "job.json"
            if not path.is_file():
                raise AppError("JOB_NOT_FOUND", "长任务清单不存在", 404)
            manifest = json.loads(path.read_text(encoding="utf-8"))
        voice = self.voices.get(manifest["voice_id"])
        request = TTSRequest(
            text=manifest["text"],
            voice_id=manifest["voice_id"],
            model=manifest.get("model", config.DEFAULT_MODEL),
            mode=manifest.get("mode", config.DEFAULT_MODE),
            client_request_id=job_id,
            quality=manifest.get("quality", config.DEFAULT_QUALITY),
            postprocess=manifest.get("postprocess", config.DEFAULT_POSTPROCESS),
            speed=manifest.get("speed"),
            pitch=manifest.get("pitch"),
            volume=manifest.get("volume"),
            emotion=manifest.get("emotion"),
        )
        job_dir = self.jobs_root / job_id
        long = LongJob(
            job_id=job_id,
            request=request,
            voice=voice,
            job_dir=job_dir,
            created_at=manifest.get("created_at") or _now(),
            segments_total=int(manifest.get("segments_total") or len(_segments_for(request, self.engine.name))),
            segments_done=int(manifest.get("segments_done") or 0),
            chars_total=int(manifest.get("chars_total") or sum(len(segment) for segment in _segments_for(request, self.engine.name))),
            chars_done=0,
            infer_sec=float(manifest.get("infer_sec") or 0.0),
            vram_peak_mb=float(manifest.get("vram_peak_mb") or 0.0),
        )
        self.long_jobs[job_id] = long
        self.long_manifests.pop(job_id, None)
        self._write_manifest(long)
        self._enqueue_long(long)
        return long.status_dict()

    def load_long_jobs(self) -> None:
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        for path in self.jobs_root.glob("*/job.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("skip corrupt job manifest %s", path)
                continue
            job_id = str(data.get("job_id") or path.parent.name)
            data["job_id"] = job_id
            if data.get("status") in {"queued", "running", "cancelling"}:
                data["status"] = "interrupted"
                data["error"] = "服务重启中断，可恢复"
                data["finished_at"] = _now()
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self.long_manifests[job_id] = data

    def _write_manifest(self, long: LongJob) -> None:
        long.job_dir.mkdir(parents=True, exist_ok=True)
        target = long.job_dir / "job.json"
        temp = long.job_dir / "job.json.tmp"
        temp.write_text(
            json.dumps(long.manifest(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temp, target)

    def cancel(self, request_id: str) -> str:
        with self._jobs_lock:
            job = self._jobs.get(request_id)
        if job is None or job.future.done():
            return "completed"
        if job.status == "queued":
            job.cancel_event.set()
            return "queued"
        if job.status == "running":
            job.cancel_event.set()
            return "between_segments"
        return "not_cancellable"

    def _loop(self) -> None:
        while not self._stopping.is_set():
            job = self._queue.get()
            try:
                if job is None:
                    return
                if job.long is not None:
                    self._process_long(job.long)
                else:
                    self._process(job)
            finally:
                self._queue.task_done()

    def _process_long(self, long: LongJob) -> None:
        if long.cancel_event.is_set():
            long.status = "cancelled"
            long.finished_at = _now()
            self._finish_long(long)
            return
        long.status = "running"
        long.started_at = _now()
        self._write_manifest(long)
        try:
            segments = _segments_for(long.request, self.engine.name)
            long.segments_total = len(segments)
            long.chars_total = sum(len(segment) for segment in segments)
            long.chars_done = 0
            self._write_manifest(long)
            task, controls = long.request.task_and_controls()
            flow_steps = config.QUALITY_FLOW_STEPS[long.request.quality]
            long.flow_steps = flow_steps
            base_infer = float(long.infer_sec)
            processing_started = time.perf_counter()
            paths = [long.job_dir / f"seg_{index + 1:04d}.wav" for index in range(len(segments))]
            existing = [
                index
                for index, path in enumerate(paths)
                if path.is_file() and path.stat().st_size > 100
            ]
            with long.progress_lock:
                long.segments_done = len(existing)
                long.chars_done = sum(len(segments[index]) for index in existing)
                self._write_manifest(long)
            pending = [index for index in range(len(segments)) if index not in set(existing)]
            if pending:
                worker_count = 1
                begin = getattr(self.engine, "begin_long_job", None)
                if callable(begin):
                    try:
                        worker_count = max(1, int(begin(long.request.model, long.request.mode) or 1))
                    except Exception:  # noqa: BLE001
                        logger.exception("parallel setup failed; running single worker")
                        worker_count = 1
                worker_count = min(worker_count, len(pending))

                def process_segment(index: int) -> None:
                    segment = segments[index]
                    segment_path = paths[index]
                    if long.cancel_event.is_set():
                        return
                    with long.progress_lock:
                        long.current_segment = max(long.current_segment, index + 1)
                        self._write_manifest(long)
                    last_error: Exception | None = None
                    for attempt in range(config.LONG_JOB_RETRIES + 1):
                        if long.cancel_event.is_set():
                            return
                        try:
                            result = self.engine.synthesize(
                                text=segment,
                                voice_wav=self.voices.audio_path(long.request.voice_id),
                                voice_text=self.voices.transcript(long.request.voice_id),
                                task=task,
                                controls=controls,
                                flow_steps=flow_steps,
                                model=long.request.model,
                                mode=long.request.mode,
                            )
                            audio.write_pcm16(segment_path, result.waveform, result.sample_rate)
                            with long.progress_lock:
                                long.segments_done += 1
                                long.chars_done += len(segment)
                                long.vram_peak_mb = max(long.vram_peak_mb, result.vram_peak_mb)
                                self._write_manifest(long)
                            last_error = None
                            break
                        except AppError as exc:
                            last_error = exc
                            logger.warning(
                                "long job %s segment %s attempt %s failed: %s",
                                long.job_id,
                                index + 1,
                                attempt + 1,
                                exc.message,
                            )
                            if exc.code in {"MODEL_NOT_READY", "VOICE_NOT_FOUND"}:
                                break
                            time.sleep(0.5)
                        except Exception as exc:  # noqa: BLE001
                            last_error = exc
                            logger.exception(
                                "long job %s segment %s attempt %s crashed",
                                long.job_id,
                                index + 1,
                                attempt + 1,
                            )
                            time.sleep(0.5)
                    if last_error is not None:
                        raise AppError(
                            "GENERATION_FAILED",
                            f"第 {index + 1}/{len(segments)} 段生成失败：{last_error}",
                            500,
                        )

                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
                        futures = [pool.submit(process_segment, index) for index in pending]
                        for future in concurrent.futures.as_completed(futures):
                            future.result()
                finally:
                    end = getattr(self.engine, "end_long_job", None)
                    if callable(end):
                        end()
            if long.cancel_event.is_set():
                long.status = "cancelled"
                long.finished_at = _now()
                self._finish_long(long)
                return

            raw_full = long.job_dir / "full_raw.wav"
            raw_duration = audio.concat_with_gap(paths, raw_full, gap_sec=config.LONG_JOB_GAP_SEC)
            date_dir = self.outputs_dir / datetime.now().strftime("%Y%m%d")
            date_dir.mkdir(parents=True, exist_ok=True)
            final_name = f"tts_{long.job_id}.wav"
            final_path = date_dir / final_name
            audio.postprocess(raw_full, final_path, long.request.postprocess)
            long.infer_sec = base_infer + (time.perf_counter() - processing_started)
            raw_full.unlink(missing_ok=True)
            long.duration_sec = audio.duration_sec(final_path)
            long.rtf = long.infer_sec / long.duration_sec if long.duration_sec > 0 else 0.0
            long.audio_url = f"/api/audio/{date_dir.name}/{final_name}"
            long.status = "done"
            long.finished_at = _now()
            self._finish_long(long)
            queue_sec = _status_from_data(long.manifest())["queue_sec"]
            self.history.append(
                HistoryItem(
                    request_id=long.job_id,
                    voice_id=long.request.voice_id,
                    text_preview=long.text_preview,
                    audio_url=long.audio_url,
                    duration_sec=round(long.duration_sec, 3),
                    elapsed_sec=round(queue_sec + long.infer_sec, 1),
                    rtf=round(long.rtf, 3),
                    quality=long.request.quality,
                    postprocess=long.request.postprocess,
                    created_at=_now(),
                )
            )
            (long.job_dir / "STAGE_DONE.json").write_text(
                json.dumps(
                    {
                        "status": "done",
                        "job_id": long.job_id,
                        "segments_total": long.segments_total,
                        "duration_sec": round(long.duration_sec, 3),
                        "audio_url": long.audio_url,
                        "finished_at": long.finished_at,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.info(
                "long job %s done: %s segments, %.1fs audio, rtf %.2f",
                long.job_id,
                long.segments_total,
                long.duration_sec,
                long.rtf,
            )
        except AppError as exc:
            long.status = "cancelled" if exc.code == "CANCELLED" else "error"
            long.error = exc.message
            long.finished_at = _now()
            self._finish_long(long)
        except Exception as exc:  # noqa: BLE001
            logger.exception("long job %s crashed", long.job_id)
            long.status = "error"
            long.error = f"{type(exc).__name__}: {exc}"
            long.finished_at = _now()
            self._finish_long(long)

    def _finish_long(self, long: LongJob) -> None:
        self._write_manifest(long)
        self.long_manifests[long.job_id] = long.manifest()

    def _process(self, job: Job) -> None:
        if job.cancel_event.is_set():
            self._finish_error(job, AppError("CANCELLED", "任务已取消", 409))
            return
        job.started_at = time.perf_counter()
        job.status = "running"
        try:
            voice_wav = self.voices.audio_path(job.request.voice_id)
            voice_text = self.voices.transcript(job.request.voice_id)
            segments = _segments_for(job.request, self.engine.name)
            task, controls = job.request.task_and_controls()
            flow_steps = config.QUALITY_FLOW_STEPS[job.request.quality]

            chunks: list[np.ndarray] = []
            peak = 0.0
            engine_seconds = 0.0
            sample_rate = self.engine.sample_rate
            for index, segment in enumerate(segments):
                if job.cancel_event.is_set():
                    raise AppError(
                        "CANCELLED",
                        f"已取消（完成 {index}/{len(segments)} 段）",
                        409,
                    )
                result = self.engine.synthesize(
                    text=segment,
                    voice_wav=voice_wav,
                    voice_text=voice_text,
                    task=task,
                    controls=controls,
                    flow_steps=flow_steps,
                    model=job.request.model,
                    mode=job.request.mode,
                )
                chunks.append(result.waveform)
                sample_rate = result.sample_rate
                peak = max(peak, result.vram_peak_mb)
                engine_seconds += result.infer_sec
                if index < len(segments) - 1:
                    chunks.append(np.zeros(int(sample_rate * config.SEGMENT_SILENCE_SEC), dtype=np.float32))

            if not chunks:
                raise AppError("GENERATION_FAILED", "没有生成任何音频", 500)
            merged = np.concatenate(chunks).astype(np.float32)
            date_dir = self.outputs_dir / datetime.now().strftime("%Y%m%d")
            date_dir.mkdir(parents=True, exist_ok=True)
            raw_path = date_dir / f"tts_{job.request_id}_raw.wav"
            final_name = f"tts_{job.request_id}.wav"
            final_path = date_dir / final_name
            audio.write_pcm16(raw_path, merged, sample_rate)
            post_started = time.perf_counter()
            audio.postprocess(raw_path, final_path, job.request.postprocess)
            post_sec = time.perf_counter() - post_started
            raw_path.unlink(missing_ok=True)

            duration = audio.duration_sec(final_path)
            infer_sec = engine_seconds + post_sec
            queue_sec = (job.started_at - job.created_at) if job.started_at else 0.0
            elapsed = queue_sec + infer_sec
            audio_url = f"/api/audio/{date_dir.name}/{final_name}"
            response = TTSResponse(
                request_id=job.request_id,
                model=job.request.model,
                mode=job.request.mode,
                audio_url=audio_url,
                duration_sec=round(duration, 3),
                queue_sec=round(queue_sec, 3),
                infer_sec=round(infer_sec, 3),
                elapsed_sec=round(elapsed, 3),
                rtf=round(infer_sec / duration, 3) if duration > 0 else 0.0,
                sample_rate=audio.SAMPLE_RATE,
                flow_steps=flow_steps,
                postprocess=job.request.postprocess,
                vram_peak_mb=round(peak, 1),
                cancel_scope="completed",
            )
            self.history.append(
                HistoryItem(
                    request_id=job.request_id,
                    voice_id=job.request.voice_id,
                    text_preview=job.request.text[:40],
                    audio_url=audio_url,
                    duration_sec=response.duration_sec,
                    elapsed_sec=response.elapsed_sec,
                    rtf=response.rtf,
                    quality=job.request.quality,
                    postprocess=job.request.postprocess,
                    created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                )
            )
            job.status = "done"
            self._finish_success(job, response)
        except AppError as exc:
            job.status = "cancelled" if exc.code == "CANCELLED" else "error"
            self._finish_error(job, exc)
        except Exception:  # noqa: BLE001
            logger.exception("job failed request_id=%s", job.request_id)
            job.status = "error"
            self._finish_error(job, AppError("GENERATION_FAILED", "生成失败，请查看日志", 500))

    def _finish_success(self, job: Job, response: TTSResponse) -> None:
        if not job.future.done():
            job.future.set_result(response)
        with self._jobs_lock:
            self._jobs.pop(job.request_id, None)

    def _finish_error(self, job: Job, error: AppError) -> None:
        error.request_id = error.request_id or job.request_id
        if not job.future.done():
            job.future.set_exception(error)
        with self._jobs_lock:
            self._jobs.pop(job.request_id, None)
