# arabizi-toolkit

عُدّة القياس لبحث «هل العرابيزي منمَّط فعلًا؟ قياسُ ثبات تمثيل الفونيمات في مدوّنة عرابيزي مصرية طبيعية، ومقارنتُه بالبيانات المستنبَطة» (د. رشا رزق أبو زيد، قسم أصول اللغة، كلية اللغة العربية، جامعة الأزهر).

Measurement toolkit and design simulation for the pre-registered study *Is Arabizi Really Standardised? Phoneme-Representation Consistency in a Naturalistic Egyptian Corpus, Compared with Elicited Data*.

This repository implements stage 3 of the revision methodology ("code and simulation"): the analysis functions specified in Chapter 2 of the protocol, the corrections listed in the analytical report, and a full re-run of the simulation study.

## Contents

| Module | What it does |
|---|---|
| `measures.py` | Dominance index, plug-in and Miller–Madow entropy, normalised entropy, shared denominator, rarefaction, per-writer equalisation. |
| `vowels.py` | Parses fully vocalised Arabic into consonants / short / long vowels, aligns it with the Arabizi form, and computes the short-vowel explicitness rate (H3). |
| `habit.py` | Habit (writer) vs convention (community) gap with a resampling null (H4). |
| `stats.py` | Phoneme-level and writer-level Wilcoxon tests, rank-biserial effect size, TOST equivalence, Benjamini–Hochberg. |
| `simulate.py` | The three-layer generative model and every simulation scenario (tables 14–23 plus three new ones). |
| `scripts/run_simulations.py` | Runs all scenarios from one seed and writes `results/`. |

## Changes relative to the code printed in the protocol

1. **`norm_entropy` with a denominator of one** returned a division by zero; it now returns 0 (a phoneme with a single variant is perfectly consistent). A denominator smaller than the number of observed variants is refused.
2. **Sample-size bias in H1.** The plug-in entropy estimator is biased downward in small samples, so an elicited corpus with fewer tokens looks more consistent than it is, which pushes the result toward H1. `phoneme_entropies(..., method="rarefy")` equalises sizes by subsampling; `method="mm"` applies the Miller–Madow correction. See `new_size_imbalance`.
3. **Writer-level test.** In the self-elicitation path each writer is observed in both conditions. `writer_level_test` compares writers directly, after `equalize_pair` has made token counts equal. Without equalisation the test is invalid, and `require_equal_counts` refuses unequal inputs. See `new_writer_level`.
4. **Vowel alignment** (`short_vowel_positions` and `has_latin_vowel` were undefined in the protocol) is now implemented as a minimum-cost alignment. It proposes; the annotator confirms (layer 3b).
5. **Equivalence testing** (`tost_paired`) so that a non-significant H1 result can be read as "no meaningful difference" when it falls inside a pre-registered margin.
6. **Habit/convention null model** by resampling writers from the pooled distribution (`gap_verdict`).

## Open decision for the researcher: the long-vowel convention

The protocol says long vowels do not count as short-vowel slots, and its example `kitab` treats the *a* after ت (written fatha + alif) as a long vowel. But the worked sentence in Chapter 1 (`el 7aga di sa3ba 2awi`) counts the fatha in حَا and the kasra in وِي as slots and reaches 7/7. The two examples cannot both hold. This toolkit follows the first rule (a short vowel followed by its own long-vowel letter is one long vowel), which gives 5/5 for that sentence. The chosen convention must be written into the annotation guide before deposit; `test_14_protocol_sentence` records the current choice.

## Usage

```bash
pip install -e .[test]
pytest                                   # 23 known-answer tests
python scripts/run_simulations.py        # all scenarios, seed 20260926
python scripts/run_simulations.py --only new_size_imbalance --reps 200
```

## Reproducibility

- Master seed `20260926`; each scenario gets a child generator from `numpy.random.SeedSequence`, so one scenario can be re-run alone.
- `results/environment.txt` records the Python, NumPy and SciPy versions used for the published numbers.
- Distributions in the simulation are the ranges stated in the protocol (group A: dominant 0.89–0.94; group B: 0.57–0.74; zero representation 6% natural, 0.5% elicited). They are assumptions about the design, not estimates of Egyptian Arabizi. Numbers from `results/` must never be reported as findings about the phenomenon.

## License

Code: MIT. The corpus itself is not in this repository and will be released separately under a data-use agreement.
