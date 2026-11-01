# -*- coding: utf-8 -*-
"""GPT-SoVITS 本地推理边车：单模型串行、质量闸门、HTTP 供主服务调用。"""
import argparse
import io
import json
import os
import re
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.environ.get("GPTSOVITS_PKG", os.path.join(REPO_ROOT, "models", "GPT-SoVITS", "GPT-SoVITS-v2pro-20250604"))
WPATH = os.environ.get("GPTSOVITS_GPT", os.path.join(PKG, "GPT_weights_v2ProPlus", "custom.ckpt"))
SPATH = os.environ.get("GPTSOVITS_SOVITS", os.path.join(PKG, "GPT_SoVITS", "pretrained_models", "v2Pro", "s2Gv2ProPlus.pth"))
BASE_WPATH = os.environ.get("GPTSOVITS_BASE_GPT", os.path.join(PKG, r"GPT_SoVITS\pretrained_models\s1v3.ckpt"))
BERT = os.path.join(PKG, r"GPT_SoVITS\pretrained_models\chinese-roberta-wwm-ext-large")
HUBERT = os.path.join(PKG, r"GPT_SoVITS\pretrained_models\chinese-hubert-base")

os.chdir(PKG)
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "GPT_SoVITS"))
sys.path.insert(0, os.path.join(PKG, "GPT_SoVITS", "eres2net"))

import numpy as np
import soundfile as sf
from opencc import OpenCC

ASR_DIR = os.environ.get("GPTSOVITS_ASR", "disabled")
_CC = OpenCC("t2s")
_ASR_PROMPT = "以下是普通话的句子，请用简体中文转写。"

STATE = {
    "tts": None,
    "ready": False,
    "loading": True,
    "error": None,
    "lock": threading.Lock(),
    "started_at": time.time(),
    "max_text_chars": 1100,
    "model": "base",
    "model_path": BASE_WPATH,
    "shutdown": threading.Event(),
    "device": "cuda",
    "gpu": "N/A",
    "vram_peak_mb": 0.0,
    "asr": None,
    "asr_error": None,
}
RESCUE = {
    "top_k": 20,
    "top_p": 1.0,
    "temperature": 1.0,
    "repetition_penalty": 1.5,
    "sample_steps": 32,
    "batch_size": 1,
    "parallel_infer": True,
    "fragment_interval": 0.0,
    "return_fragment": True,
    "text_split_method": "cut0",
}
MODEL_LIMITS = {"base": 1100, "ft": 180}
MODEL_MAX_SEC = {"base": 300, "ft": 120}
SAFE_CHARS = {"base": 160, "ft": 120}
CHUNK_GAP_SEC = 0.12
PUNCT = "，。！？；、,.!?;"


def _audio_stats(audio, sr, text):
    x = audio.astype(np.float32) / 32768.0
    peak = float(np.abs(x).max()) if x.size else 0.0
    rms = float(np.sqrt((x ** 2).mean())) if x.size else 0.0
    duration = len(x) / float(sr) if sr else 0.0
    expected_min = max(0.5, len(text) * 0.13)
    bad = peak < 0.06 or rms < 0.006 or duration < expected_min
    return {
        "peak": round(peak, 4),
        "rms": round(rms, 4),
        "duration": round(duration, 2),
        "expected_min": round(expected_min, 2),
        "bad": bad,
    }


def _norm_text(text):
    return re.sub(r"[\s，。！？、；：“”‘’（）《》—…·,.!?;:\"'()\[\]]", "", _CC.convert(text)).lower()


def _cer(ref, hyp):
    a, b = _norm_text(ref), _norm_text(hyp)
    if not a:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / len(a)


def _get_asr():
    if ASR_DIR.strip().lower() in {"", "off", "disabled", "none"}:
        STATE["asr_error"] = "disabled"
        return None
    if STATE["asr"] is not None:
        return STATE["asr"]
    if STATE["asr_error"]:
        return None
    try:
        from faster_whisper import WhisperModel

        started = time.perf_counter()
        STATE["asr"] = WhisperModel(ASR_DIR, device=STATE["device"], compute_type="float16")
        print("ASR_READY %.1fs" % (time.perf_counter() - started), flush=True)
    except Exception as exc:  # noqa: BLE001
        STATE["asr_error"] = "%s: %s" % (type(exc).__name__, exc)
        print("ASR_FAILED %s" % STATE["asr_error"], flush=True)
        return None
    return STATE["asr"]


def _asr_check(audio, sr, text):
    if len(_norm_text(text)) < 80:
        return 0.0, len(_norm_text(text)), True
    asr = _get_asr()
    if asr is None:
        return 0.0, len(_norm_text(text)), True
    try:
        pieces, _ = asr.transcribe(
            audio.astype(np.float32) / 32768.0,
            language="zh",
            beam_size=1,
            vad_filter=False,
            initial_prompt=_ASR_PROMPT,
        )
        hyp = "".join(piece.text for piece in pieces).strip()
        src_len = len(_norm_text(text))
        asr_len = len(_norm_text(hyp))
        score = _cer(text, hyp)
        ok = score <= 0.25 and asr_len >= src_len * 0.72
        return round(score, 4), asr_len, ok
    except Exception as exc:  # noqa: BLE001
        print("ASR_CHECK_FAILED %s: %s" % (type(exc).__name__, exc), flush=True)
        return 0.0, len(_norm_text(text)), True


def _infer_once(text, ref_wav, prompt_text, prompt_lang, text_lang, seed, speed_factor):
    tts = STATE["tts"]
    t0 = time.perf_counter()
    chunks = list(
        tts.run(
            {
                "text": text,
                "text_lang": text_lang,
                "ref_audio_path": ref_wav,
                "prompt_text": prompt_text,
                "prompt_lang": prompt_lang,
                "seed": seed,
                "speed_factor": speed_factor,
                **RESCUE,
            }
        )
    )
    infer_sec = time.perf_counter() - t0
    if not chunks:
        raise RuntimeError("模型未返回音频")
    sr = int(chunks[0][0])
    audio = np.concatenate([np.asarray(chunk[1], dtype=np.int16) for chunk in chunks if len(chunk[1])])
    stats = _audio_stats(audio, sr, text)
    stats.update({"seed": seed, "infer_sec": round(infer_sec, 2)})
    return audio, sr, stats


def _infer_batch(chunks, ref_wav, prompt_text, prompt_lang, text_lang, seed, speed_factor, batch_size):
    tts = STATE["tts"]
    original = tts.text_preprocessor.pre_seg_text
    tts.text_preprocessor.pre_seg_text = lambda text, lang, method: chunks
    started = time.perf_counter()
    try:
        outputs = list(
            tts.run(
                {
                    "text": "x",
                    "text_lang": text_lang,
                    "ref_audio_path": ref_wav,
                    "prompt_text": prompt_text,
                    "prompt_lang": prompt_lang,
                    "seed": seed,
                    "speed_factor": speed_factor,
                    **RESCUE,
                    "batch_size": batch_size,
                }
            )
        )
    finally:
        tts.text_preprocessor.pre_seg_text = original
    infer_sec = time.perf_counter() - started
    if not outputs:
        raise RuntimeError("模型未返回音频")
    sr = int(outputs[0][0])
    audio = np.concatenate([np.asarray(item[1], dtype=np.int16) for item in outputs if len(item[1])])
    stats = _audio_stats(audio, sr, "".join(chunks))
    stats.update(
        {
            "seed": seed,
            "infer_sec": round(infer_sec, 2),
            "chunks": len(outputs),
            "batch": batch_size,
        }
    )
    return audio, sr, stats


def _set_model(model):
    if model == STATE["model"]:
        return
    tts = STATE["tts"]
    path = BASE_WPATH if model == "base" else WPATH
    if model != "base" and not os.path.isfile(path):
        raise FileNotFoundError(f"微调 GPT 权重不存在：{path}（请设置 GPTSOVITS_GPT 或先训练）")
    tts.init_t2s_weights(path)
    tts.configs.max_sec = MODEL_MAX_SEC[model]
    STATE["model"] = model
    STATE["model_path"] = path
    print("MODEL_SWITCH %s max_sec=%s" % (model, tts.configs.max_sec), flush=True)


def _split_text(text, limit):
    pieces = [piece.strip() for piece in re.split(rf"(?<=[{re.escape(PUNCT)}])", text) if piece.strip()]
    chunks = []
    current = ""
    for piece in pieces:
        if len(piece) > limit:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(piece), limit):
                chunks.append(piece[start : start + limit])
            continue
        if current and len(current) + len(piece) > limit:
            chunks.append(current)
            current = piece
        else:
            current += piece
    if current:
        chunks.append(current)
    return chunks or [text]


def _wav_bytes(audio, sr):
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _synth_chunk(text, ref_wav, prompt_text, prompt_lang, text_lang, speed_factor):
    attempts = []
    best = None
    for seed in (1234, -1, 42):
        audio, sr, stats = _infer_once(text, ref_wav, prompt_text, prompt_lang, text_lang, seed, speed_factor)
        stats["chars"] = len(text)
        stats["chunk"] = True
        if not stats["bad"]:
            cer_score, asr_len, ok = _asr_check(audio, sr, text)
            stats["asr_cer"] = cer_score
            stats["asr_chars"] = asr_len
            if not ok:
                stats["bad"] = True
                stats["gate"] = "asr"
        attempts.append(stats)
        if not stats["bad"]:
            return audio, sr, stats, attempts
        if stats.get("gate") == "asr" and (best is None or stats["asr_cer"] < best[2]["asr_cer"]):
            best = (audio, sr, stats)
    if best is not None:
        best[2]["accepted_bad"] = True
        return best[0], best[1], best[2], attempts
    return None, None, attempts[-1], attempts


def _synth_chunk_resilient(text, ref_wav, prompt_text, prompt_lang, text_lang, speed_factor, depth=0):
    audio, sr, stats, attempts = _synth_chunk(
        text, ref_wav, prompt_text, prompt_lang, text_lang, speed_factor
    )
    if audio is not None:
        return audio, sr, stats, attempts
    if depth < 2 and len(_norm_text(text)) >= 120:
        subchunks = _split_text(text, max(60, len(text) // 2))
        if len(subchunks) > 1:
            audios = []
            total_infer = 0.0
            all_attempts = list(attempts)
            chunk_sr = sr or 32000
            for index, sub in enumerate(subchunks):
                sub_audio, sub_sr, sub_stats, sub_attempts = _synth_chunk_resilient(
                    sub, ref_wav, prompt_text, prompt_lang, text_lang, speed_factor, depth + 1
                )
                all_attempts.extend(sub_attempts)
                if sub_audio is None:
                    return None, sub_sr, sub_stats, all_attempts
                chunk_sr = sub_sr
                total_infer += sub_stats["infer_sec"]
                audios.append(sub_audio)
                if index < len(subchunks) - 1:
                    audios.append(np.zeros(int(chunk_sr * CHUNK_GAP_SEC), dtype=np.int16))
            merged = np.concatenate(audios)
            merged_stats = _audio_stats(merged, chunk_sr, text)
            merged_stats.update(
                {
                    "infer_sec": round(total_infer, 2),
                    "chars": len(text),
                    "chunks": len(subchunks),
                    "gate": "split",
                }
            )
            if not merged_stats["bad"]:
                return merged, chunk_sr, merged_stats, all_attempts
    return None, sr, stats, attempts


def synthesize(text, ref_wav, prompt_text, prompt_lang, text_lang, speed_factor, model, batch_size=1, chunk_chars=None):
    with STATE["lock"]:
        _set_model(model)
        batch_size = max(1, min(8, int(batch_size)))
        chunks = _split_text(text, int(chunk_chars or SAFE_CHARS.get(model, 260)))
        if len(chunks) >= 2:
            for seed in (1234, -1):
                try:
                    audio, sr, stats = _infer_batch(
                        chunks, ref_wav, prompt_text, prompt_lang, text_lang, seed, speed_factor, batch_size
                    )
                except Exception as exc:  # noqa: BLE001
                    print("BATCH_FAILED %s: %s" % (type(exc).__name__, exc), flush=True)
                    break
                stats["input_chunks"] = len(chunks)
                if not stats["bad"]:
                    cer_score, asr_len, ok = _asr_check(audio, sr, text)
                    stats["asr_cer"] = cer_score
                    stats["asr_chars"] = asr_len
                    if not ok:
                        stats["bad"] = True
                        stats["gate"] = "asr"
                if not stats["bad"]:
                    STATE["vram_peak_mb"] = round(__import__("torch").cuda.max_memory_allocated() / 1024**2, 1)
                    return _wav_bytes(audio, sr), sr, stats, [stats]
        if len(chunks) == 1 and len(_norm_text(text)) >= 80:
            audio, sr, stats, attempts = _synth_chunk_resilient(
                text, ref_wav, prompt_text, prompt_lang, text_lang, speed_factor
            )
            if audio is None:
                raise RuntimeError("质量闸门未通过：" + json.dumps(attempts, ensure_ascii=False))
            STATE["vram_peak_mb"] = round(__import__("torch").cuda.max_memory_allocated() / 1024**2, 1)
            return _wav_bytes(audio, sr), sr, stats, attempts
        if len(chunks) == 1:
            attempts = []
            plans = [(1234, text), (-1, text)]
            plans.append((42, "嗯，" + text) if len(text) <= 180 else (42, text))
            for seed, work_text in plans:
                audio, sr, stats = _infer_once(work_text, ref_wav, prompt_text, prompt_lang, text_lang, seed, speed_factor)
                stats["chars"] = len(work_text)
                if not stats["bad"]:
                    cer_score, asr_len, ok = _asr_check(audio, sr, work_text)
                    stats["asr_cer"] = cer_score
                    stats["asr_chars"] = asr_len
                    if not ok:
                        stats["bad"] = True
                        stats["gate"] = "asr"
                attempts.append(stats)
                if not stats["bad"]:
                    STATE["vram_peak_mb"] = round(__import__("torch").cuda.max_memory_allocated() / 1024**2, 1)
                    return _wav_bytes(audio, sr), sr, stats, attempts
            raise RuntimeError("质量闸门未通过：" + json.dumps(attempts, ensure_ascii=False))

        audios = []
        sr = 32000
        total_infer = 0.0
        all_attempts = []
        for index, chunk in enumerate(chunks):
            audio, chunk_sr, stats, attempts = _synth_chunk_resilient(
                chunk, ref_wav, prompt_text, prompt_lang, text_lang, speed_factor
            )
            all_attempts.extend(attempts)
            if audio is None:
                raise RuntimeError(
                    "第 %d/%d 块质量闸门未通过：%s"
                    % (index + 1, len(chunks), json.dumps(attempts, ensure_ascii=False))
                )
            sr = chunk_sr
            total_infer += stats["infer_sec"]
            audios.append(audio)
            if index < len(chunks) - 1:
                audios.append(np.zeros(int(sr * CHUNK_GAP_SEC), dtype=np.int16))
        merged = np.concatenate(audios)
        stats = _audio_stats(merged, sr, text)
        stats.update(
            {
                "infer_sec": round(total_infer, 2),
                "seed": 1234,
                "chars": len(text),
                "chunks": len(chunks),
            }
        )
        if stats["bad"]:
            raise RuntimeError("合并音频质量闸门未通过：" + json.dumps(stats, ensure_ascii=False))
        STATE["vram_peak_mb"] = round(__import__("torch").cuda.max_memory_allocated() / 1024**2, 1)
        return _wav_bytes(merged, sr), sr, stats, all_attempts


def _load_model():
    try:
        import torch

        if not os.path.isfile(BASE_WPATH):
            raise FileNotFoundError(BASE_WPATH)
        if not os.path.isfile(SPATH):
            raise FileNotFoundError(SPATH)
        from TTS_infer_pack.TTS import TTS, TTS_Config

        cfg = {
            "custom": {
                "device": STATE["device"],
                "is_half": True,
                "version": "v2ProPlus",
                "t2s_weights_path": BASE_WPATH,
                "vits_weights_path": SPATH,
                "cnhuhbert_base_path": HUBERT,
                "bert_base_path": BERT,
            }
        }
        started = time.perf_counter()
        tts = TTS(TTS_Config(cfg))
        tts.configs.max_sec = MODEL_MAX_SEC["base"]
        STATE["tts"] = tts
        STATE["model"] = "base"
        STATE["model_path"] = BASE_WPATH
        STATE["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        STATE["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        STATE["ready"] = True
        print("MODEL_READY %.1fs vram=%.0fMB" % (time.perf_counter() - started, STATE["vram_peak_mb"]), flush=True)
    except Exception as exc:  # noqa: BLE001
        STATE["error"] = "%s: %s" % (type(exc).__name__, exc)
        print("MODEL_FAILED %s\n%s" % (STATE["error"], traceback.format_exc()), flush=True)
    finally:
        STATE["loading"] = False


class Handler(BaseHTTPRequestHandler):
    server_version = "GPTSoVITSSidecar/1.0"

    def log_message(self, fmt, *args):
        print("%s %s" % (self.address_string(), fmt % args), flush=True)

    def _send_json(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 2 * 1024 * 1024:
            raise ValueError("请求体为空或过大")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _health(self):
        import torch

        allocated = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        self._send_json(
            200,
            {
                "ready": STATE["ready"],
                "loading": STATE["loading"],
                "error": STATE["error"],
                "device": STATE["device"],
                "gpu": STATE["gpu"],
                "sample_rate": 32000,
                "model": STATE["model"],
                "vram_allocated_mb": round(float(allocated), 1),
                "vram_peak_mb": STATE["vram_peak_mb"],
                "uptime_sec": round(time.time() - STATE["started_at"], 1),
            },
        )

    def do_GET(self):
        if self.path == "/health":
            self._health()
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/shutdown":
            self._send_json(200, {"ok": True})
            STATE["shutdown"].set()
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if self.path != "/tts":
            self._send_json(404, {"error": "not found"})
            return
        try:
            payload = self._read_json()
            if not STATE["ready"]:
                raise RuntimeError("模型未就绪：" + str(STATE["error"] or "loading"))
            text = str(payload.get("text") or "").strip()
            ref_wav = str(payload.get("ref_wav") or "").strip()
            prompt_text = str(payload.get("prompt_text") or "").strip()
            model = str(payload.get("model") or "base").strip().lower()
            if model not in MODEL_LIMITS:
                raise ValueError("model 必须是 %s" % sorted(MODEL_LIMITS))
            limit = MODEL_LIMITS[model]
            if not text or len(text) > limit:
                raise ValueError("text 长度必须在 1..%d" % limit)
            if not ref_wav or not os.path.isfile(ref_wav):
                raise ValueError("参考音频不存在")
            if not prompt_text:
                raise ValueError("参考文本为空")
            prompt_lang = str(payload.get("prompt_lang") or "zh")
            text_lang = str(payload.get("text_lang") or "zh")
            speed = float(payload.get("speed_factor") or 1.0)
            speed = min(1.5, max(0.7, speed))
            batch_size = max(1, min(8, int(payload.get("batch_size") or 1)))
            chunk_chars = int(payload.get("chunk_chars") or 0) or None
            wav, sr, stats, attempts = synthesize(text, ref_wav, prompt_text, prompt_lang, text_lang, speed, model, batch_size, chunk_chars)
            rtf = stats["infer_sec"] / stats["duration"] if stats["duration"] else 0.0
            print(
                "TTS_OK model=%s chars=%d dur=%.2f infer=%.2fs rtf=%.2f batch=%s chunks=%s attempts=%d"
                % (model, len(text), stats["duration"], stats["infer_sec"], rtf, stats.get("batch", 1), stats.get("chunks", 1), len(attempts)),
                flush=True,
            )
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav)))
            self.send_header("X-Sample-Rate", str(sr))
            self.send_header("X-Model", model)
            self.send_header("X-Batch", str(stats.get("batch", 1)))
            self.send_header("X-Infer-Sec", "%.3f" % stats["infer_sec"])
            self.send_header("X-Rtf", "%.3f" % rtf)
            self.send_header("X-Vram-Mb", str(STATE["vram_peak_mb"]))
            self.send_header("X-Attempts", str(len(attempts)))
            self.end_headers()
            try:
                self.wfile.write(wav)
            except (BrokenPipeError, ConnectionAbortedError):
                pass
        except Exception as exc:  # noqa: BLE001
            print("TTS_FAIL %s\n%s" % (exc, traceback.format_exc()), flush=True)
            self._send_json(502, {"error": "%s: %s" % (type(exc).__name__, exc)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("GPTSOVITS_PORT", "8766")))
    parser.add_argument("--device", default=os.environ.get("GPTSOVITS_DEVICE", "cuda"))
    args = parser.parse_args()
    STATE["device"] = args.device
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("LISTEN http://%s:%d pkg=%s" % (args.host, args.port, PKG), flush=True)
    _load_model()
    try:
        while not STATE["shutdown"].wait(1.0):
            pass
    except KeyboardInterrupt:
        pass
    server.shutdown()
    print("STOPPED", flush=True)


if __name__ == "__main__":
    main()
