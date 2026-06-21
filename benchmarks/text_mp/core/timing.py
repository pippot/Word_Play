from __future__ import annotations

import os


BENCHMARK_TIMING_BASE_STEPS = 500
BENCHMARK_STEPS = max(1, int(os.environ.get("WORD_PLAY_BENCHMARK_STEPS", "40")))


def normalized_steps(steps: int, *, minimum: int = 1) -> int:
    return max(minimum, round(steps * BENCHMARK_STEPS / BENCHMARK_TIMING_BASE_STEPS))


def normalized_probability(probability: float) -> float:
    if probability <= 0:
        return 0.0
    if probability >= 1:
        return 1.0
    return 1 - (1 - probability) ** (BENCHMARK_TIMING_BASE_STEPS / BENCHMARK_STEPS)

