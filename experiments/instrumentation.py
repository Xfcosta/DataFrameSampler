from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class InstrumentedCall:
    value: T
    seconds: float
    peak_memory_mb: float


def measure_call(function: Callable[[], T]) -> InstrumentedCall[T]:
    """Measure wall time and peak Python allocations for one callable."""
    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    start = time.perf_counter()
    try:
        value = function()
        current, peak = tracemalloc.get_traced_memory()
    finally:
        seconds = time.perf_counter() - start
        if not already_tracing:
            tracemalloc.stop()
    del current
    return InstrumentedCall(
        value=value,
        seconds=seconds,
        peak_memory_mb=peak / (1024 * 1024),
    )
