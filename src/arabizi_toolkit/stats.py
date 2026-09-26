"""Pre-registered tests: phoneme-level and writer-level comparisons, equivalence, FDR."""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

from .errors import ProtocolError

MIN_PHONEMES_FOR_OMNIBUS = 6


def phoneme_level_test(h_nat, h_eli) -> dict:
    """Paired Wilcoxon signed-rank over phonemes (H1, secondary in the self-elicitation path).

    Refuses fewer than six shared phonemes: with four pairs the smallest attainable
    two-sided p is 0.125, with five it is 0.0625, so the test cannot reject at 0.05.
    """
    h_nat, h_eli = np.asarray(h_nat, float), np.asarray(h_eli, float)
    if len(h_nat) != len(h_eli):
        raise ValueError("paired vectors must have equal length")
    if len(h_nat) < MIN_PHONEMES_FOR_OMNIBUS:
        raise ProtocolError(f"omnibus test needs >= {MIN_PHONEMES_FOR_OMNIBUS} shared phonemes")
    res = wilcoxon(h_nat, h_eli)
    return {"n": len(h_nat), "statistic": float(res.statistic), "p": float(res.pvalue),
            "rank_biserial": rank_biserial(h_nat - h_eli),
            "mean_diff": float(np.mean(h_nat - h_eli))}


def writer_level_test(w_nat, w_eli) -> dict:
    """Paired Wilcoxon over writers (primary test in the self-elicitation path).

    Inputs are each writer's mean normalised entropy in the two conditions, computed
    after equalising token counts per writer and phoneme (measures.equalize_pair).
    """
    w_nat, w_eli = np.asarray(w_nat, float), np.asarray(w_eli, float)
    keep = ~(np.isnan(w_nat) | np.isnan(w_eli))
    w_nat, w_eli = w_nat[keep], w_eli[keep]
    res = wilcoxon(w_nat, w_eli)
    return {"n": int(keep.sum()), "statistic": float(res.statistic), "p": float(res.pvalue),
            "rank_biserial": rank_biserial(w_nat - w_eli),
            "mean_diff": float(np.mean(w_nat - w_eli))}


def rank_biserial(d) -> float:
    """Matched-pairs rank-biserial correlation (zero differences dropped)."""
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    pos, neg = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((pos - neg) / ranks.sum())


def tost_paired(x, y, margin: float, alpha: float = 0.05) -> dict:
    """Two one-sided Wilcoxon tests for equivalence of paired samples within +/- margin.

    Equivalence is declared when both one-sided tests reject at alpha.
    """
    d = np.asarray(x, float) - np.asarray(y, float)
    p_lower = wilcoxon(d + margin, alternative="greater").pvalue
    p_upper = wilcoxon(d - margin, alternative="less").pvalue
    p = float(max(p_lower, p_upper))
    return {"p": p, "equivalent": p < alpha, "margin": margin}


def benjamini_hochberg(pvals, q: float = 0.05):
    """Return (adjusted p-values, reject flags) under Benjamini-Hochberg FDR control."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(adj_sorted, 1.0)
    return adj, adj <= q
