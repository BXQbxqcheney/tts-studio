from __future__ import annotations

import pytest

from app.errors import AppError
from app.services.text import int_to_cn, normalize_text, segment_text


def test_number_readings():
    assert int_to_cn(0) == "零"
    assert int_to_cn(10) == "十"
    assert int_to_cn(11) == "十一"
    assert int_to_cn(2026) == "二千零二十六"
    assert int_to_cn(10001) == "一万零一"
    assert int_to_cn(123456789) == "一亿二千三百四十五万六千七百八十九"


def test_date_time_money_percent_normalization():
    text = normalize_text("我们约在2026年10月31日晚上7:30，价格368元，涨幅50%。")
    assert "二零二六年十月三十一日" in text
    assert "七点半" in text
    assert "三百六十八元" in text
    assert "百分之五十" in text


def test_paralinguistic_tag_preserved():
    text = normalize_text("[Laughter]今天有2026个人。")
    assert "[Laughter]" in text
    assert "二千零二十六" in text


def test_500_chars_without_punctuation_is_segmented():
    segments = segment_text("无" * 500)
    assert len(segments) == 7
    assert max(len(segment) for segment in segments) <= 80
    assert all(segment.strip() for segment in segments)


def test_long_text_is_segmented():
    segments = segment_text("无" * 1000)
    assert len(segments) == 13
    assert max(len(segment) for segment in segments) <= 80


def test_text_too_long():
    with pytest.raises(AppError) as exc:
        segment_text("无" * 10_001)
    assert exc.value.code == "TEXT_TOO_LONG"


def test_empty_text():
    with pytest.raises(AppError):
        segment_text("   ")


def test_short_tail_is_merged_into_previous_segment():
    segments = segment_text("无" * 79 + "。好。")
    assert len(segments) == 1
    assert len(segments[0]) == 82


def test_short_first_piece_is_merged_forward():
    segments = segment_text("你好。" + "无" * 79 + "。")
    assert len(segments) == 1
    assert len(segments[0]) == 83


def test_gpt_long_segments_use_configured_max():
    segments = segment_text("无" * 2500, max_chars=1000, min_chars=20)
    assert len(segments) == 3
    assert max(len(segment) for segment in segments) <= 1000
