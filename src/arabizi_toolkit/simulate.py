"""Generative model and simulation scenarios for design sensitivity.

Model (Chapter 2, section 5 of the protocol):
  community distribution p_c per phoneme (last cell = zero representation)
  writer distribution   p_w ~ Dirichlet(alpha * p_c)
  tokens                c_w ~ Multinomial(n_w, p_w)
Elicitation effect, applied to affected phonemes only:
  concentration p_e ∝ p^(1 + delta), and the zero-representation share drops
  from its natural value to `zero_eli`.

Every scenario takes a numpy Generator, so results are reproducible from a seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import chi2_contingency, wilcoxon

from .measures import equalize_pair, miller_madow, norm_entropy, rarefy
from .stats import MIN_PHONEMES_FOR_OMNIBUS, tost_paired

# Group A: one dominant variant (0.89-0.94) with two or three rivals.
# Group B: dominant 0.57-0.74 with four or five rivals.
# The last cell of every vector is the zero representation (deletion).
GROUP_A = [[.92, .04, .02, .02], [.90, .05, .03, .02], [.94, .03, .01, .02],
           [.89, .06, .02, .01, .02], [.91, .05, .02, .02]]
GROUP_B = [[.60, .14, .09, .06, .05, .06], [.70, .11, .07, .05, .01, .06],
           [.57, .19, .09, .06, .03, .06], [.74, .09, .06, .03, .02, .06],
           [.65, .14, .08, .04, .03, .06]]


def _norm(v):
    v = np.asarray(v, float)
    return v / v.sum()


PHONEMES = [_norm(p) for p in GROUP_A + GROUP_B]


@dataclass
class Design:
    writers: int = 50
    nat_tokens: int = 15          # tokens per writer per phoneme, natural corpus
    eli_tokens: int = 15          # tokens per writer per phoneme, elicited corpus
    alpha: float = 40.0           # writer homogeneity (Dirichlet concentration)
    delta: float = 1.0            # concentration of the elicited distribution
    zero_eli: float = 0.005       # zero-representation share under elicitation
    affected: tuple = tuple(range(10))
    phonemes: list = field(default_factory=lambda: list(PHONEMES))


def elicit(p: np.ndarray, delta: float, zero_eli: float) -> np.ndarray:
    core = p[:-1] ** (1 + delta)
    core = core / core.sum() * (1 - zero_eli)
    return np.append(core, zero_eli)


def draw_corpus(d: Design, rng: np.random.Generator, condition: str):
    """Return list over phonemes of (writers x variants) count arrays."""
    out = []
    for i, p in enumerate(d.phonemes):
        n = d.nat_tokens if condition == "nat" else d.eli_tokens
        rows = []
        for _ in range(d.writers):
            pw = rng.dirichlet(d.alpha * p)
            if condition == "eli" and i in d.affected:
                pw = elicit(pw, d.delta, d.zero_eli)
            rows.append(rng.multinomial(n, pw))
        out.append(np.array(rows))
    return out


def draw_paired(d: Design, rng: np.random.Generator):
    """Self-elicitation path: the same writers (same habits) in both conditions."""
    nat, eli = [], []
    for i, p in enumerate(d.phonemes):
        rn, re_ = [], []
        for _ in range(d.writers):
            pw = rng.dirichlet(d.alpha * p)
            pe = elicit(pw, d.delta, d.zero_eli) if i in d.affected else pw
            rn.append(rng.multinomial(d.nat_tokens, pw))
            re_.append(rng.multinomial(d.eli_tokens, pe))
        nat.append(np.array(rn))
        eli.append(np.array(re_))
    return nat, eli


def _k(a, b):
    return max(int(np.count_nonzero(a + b)), 1)


def phoneme_entropies(nat, eli, method="plugin", rng=None, drop_zero=False):
    """Per-phoneme normalised entropies for the two corpora (pooled over writers)."""
    hn, he = [], []
    for cn, ce in zip(nat, eli):
        a, b = cn.sum(axis=0), ce.sum(axis=0)
        if drop_zero:
            a, b = a[:-1], b[:-1]
        if method == "rarefy":
            m = int(min(a.sum(), b.sum()))
            big_is_nat = a.sum() >= b.sum()
            big, small = (a, b) if big_is_nat else (b, a)
            subs = [rarefy(big, m, rng) for _ in range(20)]
            k = max(max(_k(s, small) for s in subs), 1)
            h_big = np.mean([norm_entropy(s, k) for s in subs])
            h_small = norm_entropy(small, k)
            x, y = (h_big, h_small) if big_is_nat else (h_small, h_big)
        else:
            k = _k(a, b)
            corrected = method == "mm"
            x, y = norm_entropy(a, k, corrected), norm_entropy(b, k, corrected)
        hn.append(x)
        he.append(y)
    return np.array(hn), np.array(he)


def _reject(hn, he):
    if len(hn) < MIN_PHONEMES_FOR_OMNIBUS:
        return False
    return wilcoxon(hn, he).pvalue < 0.05


# ---------------------------------------------------------------- scenarios

def type1_by_phonemes(rng, reps=500):
    rows = []
    for k in (4, 6, 8, 10):
        d = Design(delta=0.0, zero_eli=0.06, affected=(), phonemes=PHONEMES[:k])
        rej = 0
        for _ in range(reps):
            hn, he = phoneme_entropies(draw_corpus(d, rng, "nat"), draw_corpus(d, rng, "eli"))
            if k < MIN_PHONEMES_FOR_OMNIBUS:
                rej += wilcoxon(hn, he).pvalue < 0.05
            else:
                rej += _reject(hn, he)
        rows.append({"phonemes": k, "type1": rej / reps})
    return rows


def power_by_affected(rng, reps=500, delta=0.3):
    rows = []
    for n_aff in (1, 2, 3, 4, 5, 7, 10):
        aff = tuple(range(10 - n_aff, 10))   # affect group B first, then A
        d = Design(delta=delta, affected=aff)
        rej, diff = 0, []
        for _ in range(reps):
            hn, he = phoneme_entropies(draw_corpus(d, rng, "nat"), draw_corpus(d, rng, "eli"))
            rej += _reject(hn, he)
            diff.append(np.mean(hn - he))
        rows.append({"affected": n_aff, "mean_diff": float(np.mean(diff)), "power": rej / reps})
    return rows


def power_by_included(rng, reps=500, delta=0.3):
    rows = []
    for k in (4, 6, 8, 10):
        idx = list(range(5 - k // 2, 5 + k // 2))           # half A, half B
        phon = [PHONEMES[i] for i in idx]
        aff = tuple(range(k // 2, k))                       # the B half is affected
        d = Design(delta=delta, affected=aff, phonemes=phon)
        rej = sum(_reject(*phoneme_entropies(draw_corpus(d, rng, "nat"), draw_corpus(d, rng, "eli")))
                  for _ in range(reps))
        rows.append({"included": k, "affected": k // 2, "power": rej / reps})
    return rows


def zero_exclusion(rng, reps=500):
    d = Design(affected=tuple(range(10)))
    inc_h, exc_h, inc_d, exc_d = [], [], [], []
    for _ in range(reps):
        nat, eli = draw_corpus(d, rng, "nat"), draw_corpus(d, rng, "eli")
        a, b = phoneme_entropies(nat, eli)
        c, e = phoneme_entropies(nat, eli, drop_zero=True)
        inc_h.append(a.mean()); exc_h.append(c.mean())
        inc_d.append((a - b).mean()); exc_d.append((c - e).mean())
    return [{"indicator": "mean_H_natural", "with_zero": float(np.mean(inc_h)), "without_zero": float(np.mean(exc_h))},
            {"indicator": "difference_nat_minus_eli", "with_zero": float(np.mean(inc_d)), "without_zero": float(np.mean(exc_d))}]


def habit_gap_by_alpha(rng, reps=200):
    from .habit import habit_convention_gap
    rows = []
    for alpha in (5, 15, 40, 100, 400):
        d = Design(alpha=alpha)
        hw, hp, raw_gap = [], [], []
        for _ in range(reps):
            nat = draw_corpus(d, rng, "nat")
            for cn in nat:
                g = habit_convention_gap(cn, _k(cn.sum(axis=0), cn.sum(axis=0)), corrected=True)
                hw.append(g["habit"]); hp.append(g["convention"]); raw_gap.append(g["gap"])
        rows.append({"alpha": alpha, "habit_H": float(np.nanmean(hw)), "convention_H": float(np.nanmean(hp)),
                     "gap": float(np.nanmean(raw_gap))})
    return rows


def group_difference(rng, reps=200):
    d = Design()
    a, b = [], []
    for _ in range(reps):
        hn, _ = phoneme_entropies(draw_corpus(d, rng, "nat"), draw_corpus(d, rng, "nat"))
        a.append(hn[:5].mean()); b.append(hn[5:].mean())
    return [{"group": "A", "mean_H": float(np.mean(a))}, {"group": "B", "mean_H": float(np.mean(b))},
            {"group": "B-A", "mean_H": float(np.mean(b) - np.mean(a))}]


def power_by_tokens(rng, reps=500, delta=0.3):
    rows = []
    for n in (5, 10, 15, 25, 40):
        d = Design(nat_tokens=n, eli_tokens=n, delta=delta, affected=tuple(range(5, 10)))
        rej = sum(_reject(*phoneme_entropies(draw_corpus(d, rng, "nat"), draw_corpus(d, rng, "eli")))
                  for _ in range(reps))
        rows.append({"tokens_per_writer": n, "power": rej / reps})
    return rows


def power_by_homogeneity(rng, reps=500, delta=0.3):
    rows = []
    for alpha in (5, 15, 40, 100, 400):
        d = Design(alpha=alpha, delta=delta, affected=tuple(range(5, 10)))
        rej = sum(_reject(*phoneme_entropies(draw_corpus(d, rng, "nat"), draw_corpus(d, rng, "eli")))
                  for _ in range(reps))
        rows.append({"alpha": alpha, "power": rej / reps})
    return rows


def pooled_chisquare_type1(rng, reps=500):
    d = Design(delta=0.0, zero_eli=0.06, affected=())
    rej_ph = rej_chi = 0
    for _ in range(reps):
        nat, eli = draw_corpus(d, rng, "nat"), draw_corpus(d, rng, "eli")
        rej_ph += _reject(*phoneme_entropies(nat, eli))
        # the common but invalid alternative: chi-square on pooled token counts, one phoneme
        a, b = nat[5].sum(axis=0), eli[5].sum(axis=0)
        keep = (a + b) > 0
        rej_chi += chi2_contingency(np.vstack([a[keep], b[keep]]))[1] < 0.05
    tokens = d.writers * d.nat_tokens * 2
    return [{"test": "phoneme-level Wilcoxon (10 phonemes)", "type1": rej_ph / reps},
            {"test": f"pooled chi-square ({tokens} tokens, one phoneme)", "type1": rej_chi / reps}]


def vowel_control_power(rng, reps=500, writers=50, slots=60):
    """Paired writer-level test of explicitness in deletable vs non-deletable slots."""
    rows = []
    for diff in (0.02, 0.04, 0.06, 0.08, 0.12):
        rej = 0
        for _ in range(reps):
            base = rng.beta(18, 3, writers)             # writer rates around 0.86
            r1 = rng.binomial(slots, np.clip(base - diff, 0, 1)) / slots
            r2 = rng.binomial(slots, base) / slots
            if np.all(r1 == r2):
                continue
            rej += wilcoxon(r1, r2).pvalue < 0.05
        rows.append({"rate_difference": diff, "power": rej / reps})
    return rows


# ------------------------------------------------------ new scenarios (revision)

def size_imbalance_type1(rng, reps=1000):
    """Card 3/1: unequal token counts create a spurious difference toward H1."""
    rows = []
    for nat_n, eli_n in ((6, 6), (6, 1), (10, 1)):
        d = Design(nat_tokens=nat_n, eli_tokens=eli_n, delta=0.0, zero_eli=0.06, affected=())
        for method in ("plugin", "mm", "rarefy"):
            rej, diffs = 0, []
            for _ in range(reps):
                hn, he = phoneme_entropies(draw_corpus(d, rng, "nat"), draw_corpus(d, rng, "eli"),
                                           method=method, rng=rng)
                rej += _reject(hn, he)
                diffs.append(np.mean(hn - he))
            rows.append({"nat_tokens": nat_n, "eli_tokens": eli_n, "method": method,
                         "type1": rej / reps, "mean_spurious_diff": float(np.mean(diffs))})
    return rows


def writer_vs_phoneme_level(rng, reps=400):
    """Card 3/2: writer-level paired test in the self-elicitation path."""
    rows = []
    settings = [("matched", 4, 4, 0.0, ()), ("matched", 4, 4, 0.3, tuple(range(5, 10))),
                ("matched", 4, 4, 0.5, tuple(range(5, 10))), ("matched", 4, 4, 0.3, tuple(range(10))),
                ("unmatched", 8, 4, 0.0, ())]
    for label, nn, ne, delta, aff in settings:
        d = Design(nat_tokens=nn, eli_tokens=ne, delta=delta, affected=aff,
                   zero_eli=0.06 if not aff else 0.005)
        rp = rw = 0
        for _ in range(reps):
            nat, eli = draw_paired(d, rng)
            rp += _reject(*phoneme_entropies(nat, eli, method="mm"))
            wn = np.zeros(d.writers)
            we = np.zeros(d.writers)
            for cn, ce in zip(nat, eli):
                k = len(cn[0])
                for w in range(d.writers):
                    a, b = cn[w], ce[w]
                    if label == "matched":
                        a, b = equalize_pair(a, b, rng)
                    wn[w] += miller_madow(a) / np.log2(k) if a.sum() else 0
                    we[w] += miller_madow(b) / np.log2(k) if b.sum() else 0
            rw += wilcoxon(wn, we).pvalue < 0.05
        rows.append({"counts": label, "delta": delta, "affected": len(aff),
                     "phoneme_level": rp / reps, "writer_level": rw / reps})
    return rows


def tost_power(rng, reps=500, margin=0.05):
    """Card 3/8: probability of declaring equivalence when the true difference is zero."""
    d = Design(delta=0.0, zero_eli=0.06, affected=())
    hits = 0
    for _ in range(reps):
        hn, he = phoneme_entropies(draw_corpus(d, rng, "nat"), draw_corpus(d, rng, "eli"))
        hits += tost_paired(hn, he, margin)["equivalent"]
    return [{"margin": margin, "phonemes": 10, "p_declare_equivalence": hits / reps}]


SCENARIOS = {
    "t14_type1_by_phonemes": type1_by_phonemes,
    "t15_power_by_affected": power_by_affected,
    "t16_power_by_included": power_by_included,
    "t17_zero_exclusion": zero_exclusion,
    "t18_habit_gap_by_alpha": habit_gap_by_alpha,
    "t19_group_difference": group_difference,
    "t20_power_by_tokens": power_by_tokens,
    "t21_power_by_homogeneity": power_by_homogeneity,
    "t22_pooled_chisquare": pooled_chisquare_type1,
    "t23_vowel_control_power": vowel_control_power,
    "new_size_imbalance": size_imbalance_type1,
    "new_writer_level": writer_vs_phoneme_level,
    "new_tost": tost_power,
}
