"""文本规范化与安全切段。

Ming 上游在 text_normalization=False 时会跳过 pynini 数字规范化；这里补齐
中文数字、日期、时间、金额、百分比与标点处理，并按 80 字上限强制切段，
避免无标点长串触发上游空 torch.cat。
"""

from __future__ import annotations

import logging
import re

from app import config
from app.errors import AppError

logger = logging.getLogger("tts_studio.text")

CN_DIGITS = "零一二三四五六七八九"
CN_UNITS = ("", "十", "百", "千")
CN_GROUP_UNITS = ("", "万", "亿", "万亿")
_CN_DIGIT_MAP = {str(index): char for index, char in enumerate(CN_DIGITS)}
_PUNCT = "。！？；：，、.!?;:,"
_TAG_RE = re.compile(r"(\[[^\]]+\])")


def _four_to_cn(number: int) -> str:
    parts: list[str] = []
    zero_pending = False
    for index in range(3, -1, -1):
        base = 10**index
        digit = number // base
        number %= base
        if digit:
            if zero_pending and parts:
                parts.append("零")
            parts.append(CN_DIGITS[digit] + CN_UNITS[index])
            zero_pending = False
        elif parts and number:
            zero_pending = True
    return "".join(parts)


def int_to_cn(number: int) -> str:
    if number == 0:
        return "零"
    if number < 0:
        return "负" + int_to_cn(-number)
    groups: list[int] = []
    while number:
        groups.append(number % 10000)
        number //= 10000
    parts: list[str] = []
    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        if group == 0:
            if parts and not parts[-1].endswith("零"):
                parts.append("零")
            continue
        if parts and group < 1000 and not parts[-1].endswith("零"):
            parts.append("零")
        parts.append(_four_to_cn(group) + CN_GROUP_UNITS[index])
    result = "".join(parts).replace("零零", "零").rstrip("零")
    if result.startswith("一十"):
        result = result[1:]
    return result or "零"


def _token_to_cn(token: str) -> str:
    value = token.replace(",", "")
    if "." not in value:
        return int_to_cn(int(value))
    left, right = value.split(".", 1)
    digits = "".join(_CN_DIGIT_MAP.get(char, "零") for char in right)
    return f"{int_to_cn(int(left))}点{digits}"


def _year_to_cn(value: str) -> str:
    return "".join(_CN_DIGIT_MAP.get(char, char) for char in value)


def _time_to_cn(hour: str, minute: str) -> str:
    h = int(hour)
    m = int(minute)
    if m == 0:
        return f"{int_to_cn(h)}点整"
    if m == 30:
        return f"{int_to_cn(h)}点半"
    if m < 10:
        return f"{int_to_cn(h)}点零{int_to_cn(m)}分"
    return f"{int_to_cn(h)}点{int_to_cn(m)}分"


def _normalize_segment(text: str) -> str:
    text = re.sub(
        r"(\d{4})\s*[年\-/]\s*(\d{1,2})\s*[月\-/]\s*(\d{1,2})\s*日?",
        lambda m: f"{_year_to_cn(m.group(1))}年{int_to_cn(int(m.group(2)))}月{int_to_cn(int(m.group(3)))}日",
        text,
    )
    text = re.sub(
        r"(\d{1,2})\s*月\s*(\d{1,2})\s*日",
        lambda m: f"{int_to_cn(int(m.group(1)))}月{int_to_cn(int(m.group(2)))}日",
        text,
    )
    text = re.sub(
        r"(\d{1,2})\s*[:：]\s*(\d{2})",
        lambda m: _time_to_cn(m.group(1), m.group(2)),
        text,
    )
    text = re.sub(
        r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*%",
        lambda m: "百分之" + _token_to_cn(m.group(1)),
        text,
    )
    text = re.sub(
        r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(元|块|角|分)",
        lambda m: _token_to_cn(m.group(1)) + m.group(2),
        text,
    )
    text = re.sub(
        r"\d+(?:,\d{3})*(?:\.\d+)?",
        lambda m: _token_to_cn(m.group(0)),
        text,
    )
    return text


def normalize_text(text: str) -> str:
    if not text:
        return ""
    # 全角数字/字母转半角；中文标点保留，避免破坏韵律。
    for index in range(10):
        text = text.replace(chr(0xFF10 + index), str(index))
    for index in range(26):
        text = text.replace(chr(0xFF21 + index), chr(ord("A") + index))
        text = text.replace(chr(0xFF41 + index), chr(ord("a") + index))
    text = text.replace("\u3000", " ").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    # 保留 [Laughter] 等标签原样，只规范化标签外文本。
    parts = _TAG_RE.split(text)
    normalized = []
    for part in parts:
        if part.startswith("[") and part.endswith("]"):
            normalized.append(part)
        else:
            normalized.append(_normalize_segment(part))
    return "".join(normalized).strip()


def _hard_split(piece: str, max_chars: int) -> list[str]:
    if len(piece) <= max_chars:
        return [piece]
    chunks: list[str] = []
    start = 0
    while start < len(piece):
        end = min(start + max_chars, len(piece))
        if end < len(piece):
            window = piece[start:end]
            cut = max(window.rfind(char) for char in _PUNCT)
            if cut > 0:
                end = start + cut + 1
        chunk = piece[start:end].strip()
        if chunk:
            if chunk[-1] not in _PUNCT:
                if len(chunk) >= max_chars:
                    chunk = chunk[: max_chars - 1]
                    end = start + len(chunk)
                chunk += "，"
            chunks.append(chunk)
        start = end
    return chunks


def segment_text(text: str, max_chars: int | None = None, min_chars: int | None = None) -> list[str]:
    max_chars = max(1, int(max_chars or config.MAX_SEGMENT_CHARS))
    min_chars = max(1, int(min_chars or config.MIN_SEGMENT_CHARS))
    if len(text) > config.MAX_TEXT_CHARS:
        raise AppError(
            "TEXT_TOO_LONG",
            f"文本超过 {config.MAX_TEXT_CHARS} 字符",
            413,
        )
    normalized = normalize_text(text)
    if not normalized:
        raise AppError("INVALID_REQUEST", "文本为空或无法规范化", 400)
    pieces = [part.strip() for part in re.split(rf"(?<=[{re.escape(_PUNCT)}])", normalized) if part.strip()]
    segments: list[str] = []
    current = ""
    for piece in pieces:
        if len(piece) > max_chars:
            if current:
                segments.append(current)
                current = ""
            segments.extend(_hard_split(piece, max_chars))
            continue
        if current and len(current) + len(piece) > max_chars:
            segments.append(current)
            current = piece
        else:
            current += piece
    if current:
        segments.append(current)
    segments = [segment for segment in segments if segment.strip()]
    if not segments:
        raise AppError("INVALID_REQUEST", "文本没有可合成片段", 400)
    while len(segments) > 1 and len(segments[0]) < min_chars:
        segments[1] = segments[0] + segments[1]
        segments.pop(0)
    for index in range(len(segments) - 1, 0, -1):
        if len(segments[index]) < min_chars:
            segments[index - 1] += segments[index]
            segments.pop(index)
    limit = max_chars + min_chars
    if any(len(segment) > limit for segment in segments):
        raise AppError("TEXT_SEGMENT_TOO_LONG", "存在无法切分的长片段", 413)
    return segments
