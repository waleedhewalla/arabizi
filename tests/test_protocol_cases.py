"""Known-answer tests.

Cases 1-7 are the seven cases listed in Chapter 2, section 3 of the protocol.
Cases 8 onward were added in the revision (methodology cards 3/1, 3/2, 3/6, 3/9).
"""
import numpy as np
import pytest

from arabizi_toolkit import (ProtocolError, build_k_ref, dominance, equalize_pair, miller_madow,
                             norm_entropy, raw_entropy, require_equal_counts, short_vowel_slots,
                             vowel_rate)
from arabizi_toolkit.habit import gap_verdict, habit_convention_gap
from arabizi_toolkit.simulate import Design, draw_corpus, phoneme_entropies
from arabizi_toolkit.stats import (benjamini_hochberg, phoneme_level_test, rank_biserial,
                                   tost_paired)


# ---- cases 1-7 (original protocol) -------------------------------------------------

def test_01_single_variant():
    c = np.array([100])
    assert dominance(c) == 1.0
    assert raw_entropy(c) == 0.0


def test_02_four_equal_variants():
    c = np.array([25, 25, 25, 25])
    assert dominance(c) == 0.25
    assert norm_entropy(c, 4) == pytest.approx(1.0)


def test_03_ninety_ten():
    assert raw_entropy(np.array([90, 10])) == pytest.approx(0.469, abs=5e-4)


def test_03b_worked_example_chapter1():
    # Chapter 1, section 5/3: distributions (a) and (b), shared denominator of six
    a = np.array([90, 10, 0, 0, 0, 0])
    b = np.array([90, 4, 2, 2, 1, 1])
    assert norm_entropy(a, 6) == pytest.approx(0.181, abs=5e-4)
    assert norm_entropy(b, 6) == pytest.approx(0.264, abs=5e-4)
    assert dominance(a) == dominance(b) == 0.9


def test_04_zero_tokens_is_undefined_not_zero():
    assert np.isnan(dominance(np.array([0, 0])))
    assert np.isnan(norm_entropy(np.array([0, 0]), 2))


def test_05_denominator_refuses_one_corpus():
    with pytest.raises(ProtocolError):
        build_k_ref({"ʕ": {"3": 5}}, None)


def test_06_long_vowel_not_counted():
    # كِتَاب : kasra on kaf is a short-vowel slot; fatha + alif is one long vowel
    assert short_vowel_slots("كِتَاب", "kitab") == [True]


def test_07_same_seed_same_output():
    d = Design()
    r1 = phoneme_entropies(draw_corpus(d, np.random.default_rng(7), "nat"),
                           draw_corpus(d, np.random.default_rng(8), "nat"))
    r2 = phoneme_entropies(draw_corpus(d, np.random.default_rng(7), "nat"),
                           draw_corpus(d, np.random.default_rng(8), "nat"))
    assert np.array_equal(r1[0], r2[0]) and np.array_equal(r1[1], r2[1])


# ---- cases 8-9: the denominator (card 3/9) ----------------------------------------

def test_08_denominator_of_one_returns_zero():
    # the original code divided by log2(1) = 0
    assert norm_entropy(np.array([40]), 1) == 0.0


def test_09_denominator_smaller_than_observed_is_refused():
    with pytest.raises(ProtocolError):
        norm_entropy(np.array([5, 3, 2]), 2)


def test_09b_denominator_is_union_of_both_corpora():
    k = build_k_ref({"ħ": {"7": 9, "h": 1}}, {"ħ": {"7": 5, "5": 1}})
    assert k["ħ"] == 3


# ---- cases 10-14: vowel alignment (card 3/6) --------------------------------------

def test_10_digraph_is_one_unit():
    # شَعْب : one short vowel after sh
    assert short_vowel_slots("شَعْب", "sha3b") == [True]


def test_11_expressive_lengthening_is_reduced():
    assert short_vowel_slots("حَبِيبِي", "7abibiiii") == short_vowel_slots("حَبِيبِي", "7abibi")


def test_12_deleted_vowel_is_detected():
    # مَعَ "with": two short-vowel slots
    assert short_vowel_slots("مَعَ", "ma3a") == [True, True]
    assert short_vowel_slots("مَعَ", "m3a") == [False, True]
    # مَعَايَا : only the fatha on mim is a slot; both fatha + alif pairs are long vowels
    assert short_vowel_slots("مَعَايَا", "ma3aya") == [True]


def test_13_article_vowel():
    assert short_vowel_slots("اِلْ", "el") == [True]
    assert short_vowel_slots("اِلْ", "l") == [False]


def test_14_protocol_sentence():
    # Chapter 1, section 4/5: el 7aga di sa3ba 2awi, under the long-vowel convention
    pairs = [("اِلْ", "el"), ("حَاجَة", "7aga"), ("دِي", "di"), ("صَعْبَة", "sa3ba"), ("أَوِي", "2awi")]
    assert vowel_rate(pairs) == 1.0
    total = sum(len(short_vowel_slots(a, l)) for a, l in pairs)
    assert total == 5   # the book's table counts 7; see README, "Open decision"


# ---- cases 15-17: sample size (cards 3/1, 3/2) ------------------------------------

def test_15_miller_madow_formula():
    c = np.array([8, 2])
    expected = raw_entropy(c) + (2 - 1) / (2 * 10 * np.log(2))
    assert miller_madow(c) == pytest.approx(expected)


def test_16_equalize_pair_and_refusal():
    rng = np.random.default_rng(1)
    a, b = equalize_pair(np.array([10, 5, 5]), np.array([3, 1, 0]), rng)
    assert a.sum() == b.sum() == 4
    with pytest.raises(ProtocolError):
        require_equal_counts(np.array([10, 5]), np.array([3, 1]))


def test_17_omnibus_refuses_fewer_than_six_phonemes():
    with pytest.raises(ProtocolError):
        phoneme_level_test([.2, .3, .4, .5, .6], [.1, .2, .3, .4, .5])


# ---- cases 18-21: statistics ------------------------------------------------------

def test_18_rank_biserial_extremes():
    assert rank_biserial([1, 2, 3]) == 1.0
    assert rank_biserial([-1, -2, -3]) == -1.0


def test_19_benjamini_hochberg_known_values():
    adj, rej = benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    assert np.allclose(adj, [0.04, 0.0533333, 0.0533333, 0.20], atol=1e-6)
    assert rej.tolist() == [True, False, False, False]


def test_20_tost_declares_equivalence_for_tiny_differences():
    rng = np.random.default_rng(3)
    x = rng.uniform(0.2, 0.7, 30)
    y = x + rng.normal(0, 0.005, 30)
    assert tost_paired(x, y, margin=0.05)["equivalent"]
    assert not tost_paired(x, y + 0.2, margin=0.05)["equivalent"]


def test_21_habit_gap_positive_and_null_is_not_substantial():
    rng = np.random.default_rng(5)
    p = np.array([.7, .2, .1])
    identical = np.array([rng.multinomial(20, p) for _ in range(40)])
    assert habit_convention_gap(identical, 3)["gap"] > -0.05
    assert not gap_verdict(identical, 3, rng, reps=200)["substantial"]
    distinct = np.array([rng.multinomial(20, rng.dirichlet(np.ones(3) * 0.5)) for _ in range(40)])
    assert gap_verdict(distinct, 3, rng, reps=200)["substantial"]
