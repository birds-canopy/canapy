# Licence: BSD-3-Clause
"""Acoustic segmentation of phrases into syllables.

The segmenter reads an energy threshold off each phrase with Otsu's method,
gates it with spectral flatness, and turns the resulting mask into segments.
Nothing is learned and nothing is cached.
"""
from ..utils.exceptions import AlreadySegmented
from .grouping import group_segments, period_of, phase_of
from .otsu import otsu, otsu_threshold
from .segmenter import mask_to_segments, segment_signal
from .corpus import pools_period, segmentation_params, to_syllable_level

__all__ = [
    "AlreadySegmented",
    "group_segments",
    "otsu",
    "otsu_threshold",
    "mask_to_segments",
    "period_of",
    "phase_of",
    "segment_signal",
    "pools_period",
    "segmentation_params",
    "to_syllable_level",
]
