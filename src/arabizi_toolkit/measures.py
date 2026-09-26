"""Consistency measures for phoneme-representation distributions.

A distribution is a 1-D array of non-negative integer counts, one cell per
representation variant (including the zero representation, i.e. deletion).
"""
from __future__ import annotations

import numpy as np

from .errors import ProtocolError


def _counts(counts) -> np.ndarray:
    c = np.asarray(counts)
    if c.ndim != 1:
        raise ValueError("counts must be a 1-D vector")
    if np.any(c < 0):
        raise ValueError("counts must be non-negative")
    return c


def dominance(counts) -> float:
    """Share of the most frequent variant. NaN (not 0) when there are no tokens."""
    c = _counts(counts)
    n = c.sum()
    if n == 0:
        return float("nan")
    return float(c.max() / n)


def raw_entropy(counts) -> float:
    """Plug-in Shannon entropy in bits. Zero cells are ignored."""
    c = _counts(counts)
    n = c.sum()
    if n == 0:
        return float("nan")
    p = c[c > 0] / n
    return float(-(p * np.log2(p)).sum())


def miller_madow(counts) -> float:
    """Plug-in entropy plus the Miller (1955) bias correction, in bits.

    H_MM = H + (m - 1) / (2 n ln 2), where m is the number of observed variants.
    """
    c = _counts(counts)
    n = c.sum()
    if n == 0:
        return float("nan")
    m = int(np.count_nonzero(c))
    return raw_entropy(c) + (m - 1) / (2 * n * np.log(2))


def norm_entropy(counts, k_ref: int, corrected: bool = False) -> float:
    """Entropy normalised by log2(k_ref), the shared reference denominator.

    - NaN when there are no tokens.
    - 0 when k_ref == 1: a phoneme with a single variant across both corpora is
      perfectly consistent (the original protocol code divided by log2(1) = 0).
    - Refuses k_ref smaller than the number of observed variants, which can only
      happen if the denominator was built from different data.
    """
    c = _counts(counts)
    observed = int(np.count_nonzero(c))
    if k_ref < 1:
        raise ProtocolError("k_ref must be at least 1")
    if observed > k_ref:
        raise ProtocolError(
            f"k_ref={k_ref} is smaller than the {observed} observed variants; "
            "the denominator must be built from both corpora after rare-variant review"
        )
    if c.sum() == 0:
        return float("nan")
    if k_ref == 1:
        return 0.0
    h = miller_madow(c) if corrected else raw_entropy(c)
    return float(h / np.log2(k_ref))


def build_k_ref(corpus_nat: dict | None, corpus_eli: dict | None) -> dict:
    """Shared normalising denominator per phoneme.

    Each corpus maps phoneme -> {variant: count}. The denominator is the number of
    variants attested in the two corpora together. Building it from one corpus is
    refused in code, not only in documentation.
    """
    if corpus_nat is None or corpus_eli is None:
        raise ProtocolError("the denominator is built from both corpora, never from one")
    k = {}
    for ph in set(corpus_nat) | set(corpus_eli):
        variants = {v for v, n in corpus_nat.get(ph, {}).items() if n > 0}
        variants |= {v for v, n in corpus_eli.get(ph, {}).items() if n > 0}
        k[ph] = max(len(variants), 1)
    return k


def rarefy(counts, m: int, rng: np.random.Generator) -> np.ndarray:
    """Draw m tokens without replacement from a count vector (hypergeometric)."""
    c = _counts(counts).astype(np.int64)
    n = int(c.sum())
    if m > n:
        raise ProtocolError(f"cannot rarefy {n} tokens down to {m}")
    if m == n:
        return c.copy()
    return rng.multivariate_hypergeometric(c, m)


def rarefied_norm_entropy(big, small_n: int, k_ref: int, rng: np.random.Generator,
                          draws: int = 200) -> float:
    """Mean normalised entropy of `draws` subsamples of size small_n from `big`."""
    vals = [norm_entropy(rarefy(big, small_n, rng), k_ref) for _ in range(draws)]
    return float(np.mean(vals))


def equalize_pair(nat, eli, rng: np.random.Generator):
    """Return (nat', eli') with equal token counts by subsampling the larger one.

    Used for writer-level comparisons, where unequal counts make the small-sample
    bias of the entropy estimator dominate the result.
    """
    nat = _counts(nat)
    eli = _counts(eli)
    m = int(min(nat.sum(), eli.sum()))
    return rarefy(nat, m, rng), rarefy(eli, m, rng)


def require_equal_counts(nat, eli) -> None:
    """Writer-level tests refuse to run on unequal token counts."""
    if int(np.sum(nat)) != int(np.sum(eli)):
        raise ProtocolError(
            "writer-level comparison requires equal token counts in both conditions; "
            "call equalize_pair first"
        )
