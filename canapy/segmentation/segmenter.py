# Licence: BSD-3-Clause
"""Turning one phrase of song into its syllables."""
import logging

import numpy as np

from .features import NPERSEG, contours
from .grouping import group_segments
from .otsu import otsu, otsu_threshold

logger = logging.getLogger("canapy")

BAND = (1500.0, 8000.0)
HOP = 0.0005
MIN_SYLLABLE_DURATION = 0.004
MIN_SILENCE_GAP = 0.001
WIENER_MAX = -0.2
ETA2_MIN = 0.5
GROUP_SYLLABLES = False


def mask_to_segments(mask, times, min_syllable_duration, min_silence_gap):
    """Boolean "there is sound here" mask -> list of (start, end).

    Merging comes before filtering: the other order would discard the two
    halves of a split syllable separately instead of gluing them back.
    """
    changes = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1

    segments = [[times[a], times[b]] for a, b in zip(starts, ends) if b > a]

    merged = []
    for start, end in segments:
        if merged and start - merged[-1][1] < min_silence_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    return [(s, e) for s, e in merged if e - s >= min_syllable_duration]


def segment_signal(
    signal,
    sr,
    min_syllable_duration=MIN_SYLLABLE_DURATION,
    min_silence_gap=MIN_SILENCE_GAP,
    wiener_max=WIENER_MAX,
    band=BAND,
    hop=HOP,
    eta2_min=ETA2_MIN,
    group_syllables=GROUP_SYLLABLES,
    period=None,
    single=None,
):
    """Syllables of one phrase, in that phrase's own time frame.

    Returns ``(segments, eta_squared)``. An empty list means the energy
    distribution was not bimodal enough to be split; the caller is expected to
    leave the phrase alone rather than treat it as silent.

    With ``group_syllables``, the segments returned are the repeated units
    rather than the acoustic elements — see ``grouping``. ``period`` and
    ``single`` let a caller impose values pooled over a whole label instead of
    letting this one phrase decide alone.
    """
    # too short to fill one analysis window
    if len(signal) < NPERSEG:
        return [], 0.0

    times, energy, energy_db, flatness = contours(signal, sr, band=band, hop=hop)

    threshold, eta_squared = otsu(energy_db)
    if eta_squared < eta2_min:
        return [], eta_squared

    n = min(len(energy_db), len(flatness))
    loud = energy_db[:n] >= threshold
    gate = flatness[:n] <= (otsu_threshold(flatness) if wiener_max is None else wiener_max)

    segments = mask_to_segments(
        loud & gate, times[:n], min_syllable_duration, min_silence_gap
    )

    if group_syllables and segments:
        segments, info = group_segments(segments, times, energy,
                                        period=period, single=single)
        if info["grouped"]:
            logger.debug(f"grouped into units of {info['period'] * 1000:.0f} ms "
                         f"({info['multiple']:.1f} elements each)")

    return segments, eta_squared
