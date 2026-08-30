from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np


NANOSECONDS_PER_SECOND = 1_000_000_000
MILLISECONDS_PER_SECOND = 1_000.0
TIME_SCHEMA_VERSION = 2


def active_len_from_duration(duration_ms: float, hz: float) -> int:
    """Map logical gesture duration to frames using preprocessing semantics."""
    duration = float(duration_ms)
    rate = float(hz)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration_ms must be finite and > 0")
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("hz must be finite and > 0")
    # Preserve preprocessing's operation order exactly.  Although
    # ``duration * rate / 1000`` is algebraically equivalent, IEEE-754 rounding
    # differs at half-frame boundaries such as 545/575 ms.
    frames = duration / MILLISECONDS_PER_SECOND * rate
    value = int(round(frames))
    return max(1, value)


def validate_active_len(value: Any) -> int:
    try:
        numeric = float(value)
    except Exception as exc:
        raise ValueError("active_len must be a positive integer") from exc
    if not math.isfinite(numeric) or numeric < 1 or not numeric.is_integer():
        raise ValueError("active_len must be a positive integer")
    return int(numeric)


def sample_period_ns(hz: float) -> int:
    rate = float(hz)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("hz must be finite and > 0")
    return int(round(NANOSECONDS_PER_SECOND / rate))


def validate_start_time_ns(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("start_time_ns must be an integer or None")
    try:
        numeric = int(value)
    except Exception as exc:
        raise ValueError("start_time_ns must be an integer or None") from exc
    if numeric < 0:
        raise ValueError("start_time_ns must be a non-negative integer")
    return numeric


def build_active_timeline(
    active_len: int,
    hz: float,
    logical_event_duration_ms: float,
    start_time_ns: Optional[int] = None,
    logical_active_len: Optional[int] = None,
    model_max_active_len: Optional[int] = None,
) -> Dict[str, Any]:
    length = validate_active_len(active_len)
    logical_ms = float(logical_event_duration_ms)
    if not math.isfinite(logical_ms) or logical_ms <= 0:
        raise ValueError("logical_event_duration_ms must be finite and > 0")
    start_ns = validate_start_time_ns(start_time_ns)
    period_ns = sample_period_ns(hz)
    expected_length = length if logical_active_len is None else validate_active_len(logical_active_len)
    # Only the caller that owns a fixed-size model window can declare clipping.
    # A duration/frame-count difference by itself may simply be sensor-clock
    # quantisation (the observed interval endpoints are not phase-aligned).
    model_capacity = (
        None if model_max_active_len is None else validate_active_len(model_max_active_len)
    )
    int64_max = int(np.iinfo(np.int64).max)
    relative_last = int((length - 1) * period_ns)
    logical_duration_ns = int(round(logical_ms * 1_000_000.0))
    buffer_duration_ns = int(round(length * NANOSECONDS_PER_SECOND / float(hz)))
    if max(relative_last, logical_duration_ns, buffer_duration_ns) > int64_max:
        raise OverflowError("active timeline exceeds int64 nanoseconds")
    if start_ns is not None and start_ns > int64_max - max(relative_last, logical_duration_ns, buffer_duration_ns):
        raise OverflowError("start_time_ns plus event timeline exceeds int64")
    relative = np.arange(length, dtype=np.int64) * np.int64(period_ns)
    timestamps = None if start_ns is None else relative + np.int64(start_ns)
    end_ns = None if start_ns is None else int(start_ns + logical_duration_ns)
    buffer_end_ns = None if start_ns is None else int(start_ns + buffer_duration_ns)
    buffer_ms = float(length * MILLISECONDS_PER_SECOND / float(hz))
    logical_buffer_ms = float(expected_length * MILLISECONDS_PER_SECOND / float(hz))
    clipped_frames = 0 if model_capacity is None else max(0, expected_length - model_capacity)
    return {
        "time_schema_version": TIME_SCHEMA_VERSION,
        "sample_period_ns": int(period_ns),
        "relative_timestamps_ns": relative,
        "timestamps_ns": timestamps,
        "start_time_ns": start_ns,
        "end_time_ns": end_ns,
        "buffer_end_time_ns": buffer_end_ns,
        "logical_event_duration_ms": logical_ms,
        "active_len": int(length),
        "logical_active_len": int(expected_length),
        "buffer_duration_ms": buffer_ms,
        # Quantization is only the duration-to-sample-grid difference.  A
        # fixed-size model may emit fewer samples than the logical event; that
        # separate loss is reported below and must not be called quantization.
        "logical_buffer_duration_ms": logical_buffer_ms,
        "duration_quantization_error_ms": float(logical_buffer_ms - logical_ms),
        "emitted_duration_delta_ms": float(buffer_ms - logical_ms),
        "clipped_buffer_duration_ms": float(
            clipped_frames * MILLISECONDS_PER_SECOND / float(hz)
        ),
        "is_partial_event": bool(clipped_frames > 0),
    }


def build_window_timeline(
    window_len: int,
    active_start_index: int,
    hz: float,
    start_time_ns: Optional[int] = None,
) -> Dict[str, Any]:
    length = validate_active_len(window_len)
    active_start = int(active_start_index)
    if active_start < 0 or active_start >= length:
        raise ValueError("active_start_index must lie inside the window")
    start_ns = validate_start_time_ns(start_time_ns)
    period_ns = sample_period_ns(hz)
    relative = (np.arange(length, dtype=np.int64) - np.int64(active_start)) * np.int64(period_ns)
    if start_ns is not None:
        first = int(start_ns + int(relative[0]))
        last = int(start_ns + int(relative[-1]))
        if first < 0 or last > int(np.iinfo(np.int64).max):
            raise OverflowError("start_time_ns plus window timeline exceeds non-negative int64")
    timestamps = None if start_ns is None else relative + np.int64(start_ns)
    return {
        "window_relative_timestamps_ns": relative,
        "window_timestamps_ns": timestamps,
        "window_start_time_ns": None if start_ns is None else int(timestamps[0]),
    }


__all__ = [
    "TIME_SCHEMA_VERSION",
    "active_len_from_duration",
    "build_active_timeline",
    "build_window_timeline",
    "sample_period_ns",
    "validate_active_len",
    "validate_start_time_ns",
]
