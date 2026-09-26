"""Habit (writer-internal) versus convention (community) consistency, H4.

The pooled distribution is the average of the writers' distributions and entropy
is concave, so the gap (pooled minus mean-of-writers) is positive by construction.
What carries information is its size relative to a null in which writers do not
differ at all. The null is built by resampling each writer's tokens from the
pooled distribution with that writer's own token count, which also reproduces the
small-sample bias of the per-writer estimates.
"""
from __future__ import annotations

import numpy as np

from .measures import norm_entropy

MIN_TOKENS_PER_WRITER = 10


def habit_convention_gap(writer_counts, k_ref: int, corrected: bool = True) -> dict:
    """writer_counts: 2-D array (writers x variants) for one phoneme."""
    wc = np.asarray(writer_counts)
    eligible = wc[wc.sum(axis=1) >= MIN_TOKENS_PER_WRITER]
    pooled = wc.sum(axis=0)
    h_pool = norm_entropy(pooled, k_ref, corrected=corrected)
    if len(eligible) == 0:
        return {"habit": float("nan"), "convention": h_pool, "gap": float("nan"), "writers": 0}
    h_w = np.mean([norm_entropy(r, k_ref, corrected=corrected) for r in eligible])
    return {"habit": float(h_w), "convention": h_pool, "gap": float(h_pool - h_w),
            "writers": int(len(eligible))}


def null_gap_distribution(writer_counts, k_ref: int, rng: np.random.Generator,
                          reps: int = 1000, corrected: bool = True) -> np.ndarray:
    """Gap values when all writers share the pooled distribution (no habits)."""
    wc = np.asarray(writer_counts)
    pooled = wc.sum(axis=0)
    p = pooled / pooled.sum()
    sizes = wc.sum(axis=1)
    out = np.empty(reps)
    for r in range(reps):
        sim = np.array([rng.multinomial(n, p) for n in sizes])
        out[r] = habit_convention_gap(sim, k_ref, corrected)["gap"]
    return out


def gap_verdict(writer_counts, k_ref: int, rng: np.random.Generator, reps: int = 1000) -> dict:
    """Pre-registered reading: 'substantial' only above the 95th percentile of the null."""
    obs = habit_convention_gap(writer_counts, k_ref)
    null = null_gap_distribution(writer_counts, k_ref, rng, reps)
    p95 = float(np.nanpercentile(null, 95))
    return {**obs, "null_p95": p95, "null_mean": float(np.nanmean(null)),
            "corrected_gap": obs["gap"] - float(np.nanmean(null)),
            "substantial": bool(obs["gap"] > p95)}
