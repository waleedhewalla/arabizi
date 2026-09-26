# Simulation results

seed `20260926` · toolkit 0.1.0 · numpy 2.4.6 · scipy 1.17.1 · python 3.11.15


## t14_type1_by_phonemes

| phonemes | type1 |
|---|---|
| 4 | 0.000 |
| 6 | 0.018 |
| 8 | 0.036 |
| 10 | 0.024 |

## t15_power_by_affected

| affected | mean_diff | power |
|---|---|---|
| 1 | 0.022 | 0.048 |
| 2 | 0.045 | 0.108 |
| 3 | 0.063 | 0.188 |
| 4 | 0.085 | 0.350 |
| 5 | 0.106 | 0.570 |
| 7 | 0.135 | 1.000 |
| 10 | 0.175 | 1.000 |

## t16_power_by_included

| included | affected | power |
|---|---|---|
| 4 | 2 | 0.000 |
| 6 | 3 | 0.134 |
| 8 | 4 | 0.300 |
| 10 | 5 | 0.578 |

## t17_zero_exclusion

| indicator | with_zero | without_zero |
|---|---|---|
| mean_H_natural | 0.453 | 0.430 |
| difference_nat_minus_eli | 0.319 | 0.299 |

## t18_habit_gap_by_alpha

| alpha | habit_H | convention_H | gap |
|---|---|---|---|
| 5 | 0.305 | 0.454 | 0.148 |
| 15 | 0.364 | 0.455 | 0.091 |
| 40 | 0.391 | 0.456 | 0.065 |
| 100 | 0.403 | 0.456 | 0.053 |
| 400 | 0.410 | 0.456 | 0.046 |

## t19_group_difference

| group | mean_H |
|---|---|
| A | 0.269 |
| B | 0.638 |
| B-A | 0.369 |

## t20_power_by_tokens

| tokens_per_writer | power |
|---|---|
| 5 | 0.546 |
| 10 | 0.556 |
| 15 | 0.584 |
| 25 | 0.604 |
| 40 | 0.566 |

## t21_power_by_homogeneity

| alpha | power |
|---|---|
| 5 | 0.492 |
| 15 | 0.632 |
| 40 | 0.592 |
| 100 | 0.576 |
| 400 | 0.612 |

## t22_pooled_chisquare

| test | type1 |
|---|---|
| phoneme-level Wilcoxon (10 phonemes) | 0.038 |
| pooled chi-square (1500 tokens, one phoneme) | 0.134 |

## t23_vowel_control_power

| rate_difference | power |
|---|---|
| 0.020 | 0.590 |
| 0.040 | 0.982 |
| 0.060 | 1.000 |
| 0.080 | 1.000 |
| 0.120 | 1.000 |

## new_size_imbalance

| nat_tokens | eli_tokens | method | type1 | mean_spurious_diff |
|---|---|---|---|---|
| 6 | 6 | plugin | 0.050 | -0.000 |
| 6 | 6 | mm | 0.042 | 0.000 |
| 6 | 6 | rarefy | 0.051 | -0.000 |
| 6 | 1 | plugin | 0.078 | 0.021 |
| 6 | 1 | mm | 0.041 | 0.006 |
| 6 | 1 | rarefy | 0.038 | 0.000 |
| 10 | 1 | plugin | 0.111 | 0.024 |
| 10 | 1 | mm | 0.051 | 0.006 |
| 10 | 1 | rarefy | 0.050 | 0.000 |

## new_writer_level

| counts | delta | affected | phoneme_level | writer_level |
|---|---|---|---|---|
| matched | 0.000 | 0 | 0.058 | 0.048 |
| matched | 0.300 | 5 | 0.540 | 1.000 |
| matched | 0.500 | 5 | 0.593 | 1.000 |
| matched | 0.300 | 10 | 1.000 | 1.000 |
| unmatched | 0.000 | 0 | 0.045 | 0.985 |

## new_tost

| margin | phonemes | p_declare_equivalence |
|---|---|---|
| 0.050 | 10 | 0.964 |
