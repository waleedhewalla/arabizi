"""Measurement toolkit for the study "Is Arabizi Really Standardised?".

Consistency measures, the shared denominator, sample-size equalisation,
short-vowel alignment, habit/convention gap, pre-registered tests, and the
simulation used to assess design sensitivity.
"""
from .errors import ProtocolError
from .measures import (build_k_ref, dominance, equalize_pair, miller_madow, norm_entropy,
                       rarefy, rarefied_norm_entropy, raw_entropy, require_equal_counts)
from .vowels import latin_units, parse_vocalised, short_vowel_slots, vowel_rate

__version__ = "0.1.0"
