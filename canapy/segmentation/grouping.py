# Licence: BSD-3-Clause
"""Grouping detected elements into the repeated unit they belong to.

On a hierarchical phrase, what repeats is a unit made of several notes
separated by real silence, so a silence detector returns the notes rather than
the units. The repetition period says how many elements make a unit and the
phase says which one opens it; the boundaries still come from the detected
onsets.
"""
import numpy as np

# the autocorrelation peaks at every multiple of the period, so a plain argmax
# locks onto 2T on phrases with long silences: accept the first peak within
# this fraction of the maximum instead
SUBHARMONIC = 0.9
# how far an onset may sit from the predicted unit start, as a fraction of the
# period
TOLERANCE = 0.35
# safety rail for signals that are not periodic at all
MIN_CONFIDENCE = 0.30
# below this many periods the phrase is one long syllable the detector cut into
# pieces, not a repetition
MIN_CYCLES = 1.6
PERIOD_MAX = 0.500
PHASE_BINS = 60
# width of the step read on the folded cycle, as a fraction of the median
# interval between onsets
PHASE_WINDOW = 0.5


def period_of(times, energy, min_period, subharmonic=SUBHARMONIC):
    """Repetition period, and the autocorrelation peak that vouches for it.

    `min_period` floors the search: on a phrase holding a few long units, the
    autocorrelation is otherwise dominated by the correlation internal to one
    element and returns the lower bound.
    """
    if len(energy) < 8:
        return np.nan, 0.0

    centered = energy - energy.mean()
    correlation = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
    correlation = correlation / np.arange(len(centered), 0, -1)
    if correlation[0] <= 0:
        return np.nan, 0.0
    correlation = correlation / correlation[0]

    step = times[1] - times[0]
    low = max(1, int(min_period / step))
    high = min(len(correlation) - 1, int(PERIOD_MAX / step))
    if high <= low + 1:
        return np.nan, 0.0

    window = correlation[low:high]
    candidates = np.flatnonzero(window >= subharmonic * window.max())
    best = low + int(candidates[0] if len(candidates) else np.argmax(window))
    return float(best * step), float(correlation[best])


def phase_of(onsets, times, energy, period, n_bins=PHASE_BINS,
             window=PHASE_WINDOW):
    """Where in the cycle the unit boundary sits: the largest step up.

    The period does not say which element opens a unit — an autocorrelation is
    invariant under a cyclic shift — and anchoring on the first detected element
    is wrong, since a phrase cut out of a song often starts mid-unit. A unit is
    taken to open where the energy that follows most exceeds the energy that
    precedes, read on the folded cycle: each cycle votes, so the answer does not
    depend on where the phrase was cut.

    The window is half the median interval between onsets, the only time scale
    the unit's own content provides.
    """
    if not np.isfinite(period) or period <= 0 or not len(energy):
        return 0.0

    onsets = np.asarray(onsets, dtype=float)
    interval = float(np.median(np.diff(onsets))) if len(onsets) > 1 else period

    bins = ((times % period) / period * n_bins).astype(int) % n_bins
    folded = np.array([energy[bins == k].mean() if (bins == k).any() else np.nan
                       for k in range(n_bins)])
    if not np.isfinite(folded).any():
        return 0.0
    folded = np.where(np.isfinite(folded), folded, np.nanmin(folded))
    decibels = 10 * np.log10(np.maximum(folded, 1e-20))

    width = int(round(window * interval / period * n_bins))
    width = int(np.clip(width, 1, n_bins // 2))

    # circular means of the window before and after each candidate phase
    cumulative = np.concatenate([[0.0], np.cumsum(np.tile(decibels, 3))])
    centre = np.arange(n_bins) + n_bins
    after = cumulative[centre + width] - cumulative[centre]
    before = cumulative[centre] - cumulative[centre - width]

    return float(np.argmax(after - before)) / n_bins * period



def estimate(segments, times, energy, subharmonic=SUBHARMONIC,
             min_cycles=MIN_CYCLES):
    """What one phrase says about its own period and its own nature.

    The period is a property of the label rather than of the phrase, so these
    estimates are meant to be pooled before use.
    """
    if len(segments) < 2:
        return {}

    onsets = np.array([a for a, _ in segments])
    ends = np.array([b for _, b in segments])
    period, confidence = period_of(times, energy,
                                   min_period=float(np.median(ends - onsets)),
                                   subharmonic=subharmonic)
    if not np.isfinite(period) or period <= 0:
        return {}

    cycles = float(times[-1]) / period
    return {"period": period, "confidence": confidence, "cycles": cycles,
            "single": cycles < min_cycles}


def pool_by_label(estimates):
    """One period per label, from every phrase carrying it.

    The median resists the octave errors that a mean would absorb.

    Only the period is pooled. Holding fewer than ``MIN_CYCLES`` periods is a
    property of a phrase, since it divides that phrase's duration, so
    ``group_segments`` decides it per phrase from the pooled period.
    """
    pooled = {}
    for label, rows in estimates.items():
        periods = [r["period"] for r in rows if np.isfinite(r.get("period", np.nan))]
        if not periods:
            continue
        pooled[label] = {"period": float(np.median(periods)),
                         "n_phrases": len(periods)}
    return pooled


def group_segments(segments, times, energy, tolerance=TOLERANCE,
                   min_confidence=MIN_CONFIDENCE, subharmonic=SUBHARMONIC,
                   min_cycles=MIN_CYCLES, period=None, single=None):
    """Merge elements into the units they belong to.

    Returns ``(units, info)``. ``info["grouped"]`` says whether anything was
    merged; a phrase whose elements already are its units comes back untouched.
    A phrase that turns out to hold no repetition at all collapses to a single
    unit, ``info["single"]`` marking the case.

    ``period`` and ``single`` override what this phrase would conclude on its
    own, and are how a caller feeds back a value pooled over every phrase of
    the same label.
    """
    if len(segments) < 2:
        return list(segments), {"grouped": False, "single": False}

    onsets = np.array([a for a, _ in segments])
    ends = np.array([b for _, b in segments])
    interval = float(np.median(np.diff(onsets)))

    imposed = period is not None
    measured, confidence = period_of(times, energy,
                                     min_period=float(np.median(ends - onsets)),
                                     subharmonic=subharmonic)
    period = measured if period is None else float(period)
    info = {"grouped": False, "single": False, "period": period,
            "measured_period": measured, "confidence": confidence,
            "pooled": period is not None and period != measured,
            "multiple": period / interval if interval > 0 else np.nan}

    if not np.isfinite(period) or period <= 0 or interval <= 0:
        return list(segments), info

    # Not enough periods to be a repetition: the phrase is one long syllable.
    # Checked before the confidence, which such a phrase scores low precisely
    # because it holds no repetition.
    cycles = float(times[-1]) / period if len(times) else np.nan
    info["cycles"] = cycles
    no_repetition = single if single is not None else (
        np.isfinite(cycles) and cycles < min_cycles)
    if no_repetition:
        info["single"] = True
        info["grouped"] = len(segments) > 1
        return [(onsets[0], ends[-1])], info

    # the confidence guards this phrase's own estimate; an imposed period does
    # not come from it, and a low confidence is when a pooled value is worth
    # the most
    if not imposed and confidence < min_confidence:
        return list(segments), info

    phase = phase_of(onsets, times, energy, period)
    info["phase"] = phase

    first = _first_unit(onsets, phase, period, tolerance)
    if first is None:
        return list(segments), info

    first = _walk_back(onsets, first, period, tolerance, interval)
    info["first"] = first

    units = [(onsets[0], ends[first - 1])] if first > 0 else []
    start = first
    while start < len(onsets):
        # re-anchoring on a real onset at every unit is what absorbs tempo
        # changes, where a rigid grid of period T would drift
        remaining = np.arange(start + 1, len(onsets))
        if not len(remaining):
            units.append((onsets[start], ends[-1]))
            break
        distances = np.abs(onsets[remaining] - (onsets[start] + period))
        if distances.min() > tolerance * period:
            units.extend(_trailing(onsets, ends, start, period))
            break
        best = int(remaining[np.argmin(distances)])
        units.append((onsets[start], ends[best - 1]))
        start = best

    # the walk is self-limiting: on a phrase whose elements already are its
    # units, targeting "onset + T" lands on the very next onset and gives the
    # input back, so no threshold is needed to decide whether to run it
    info["grouped"] = len(units) != len(segments)
    return units, info


def _trailing(onsets, ends, start, period):
    """What is left when the walk cannot find where its next unit begins.

    A unit closing a phrase is a fragment, so it cannot span a whole period:
    below that, this is the truncated last unit and merging it is right. At or
    above, the walk lost the beat rather than reached the end, and the elements
    are given back untouched — an over-segmentation is recoverable, a merge is
    not.
    """
    if ends[-1] - onsets[start] < period:
        return [(onsets[start], ends[-1])]
    return [(a, b) for a, b in zip(onsets[start:], ends[start:])]


def _walk_back(onsets, first, period, tolerance, interval):
    """The forward walk's re-anchoring, run backwards from the first unit.

    ``_first_unit`` reads its answer off a grid laid from ``phase``, the one
    thing here not anchored on a real onset, so it can swallow every onset
    before an arbitrary point into a single head unit. Stepping back one period
    at a time and snapping to a real onset can only add boundaries.

    The step is held to half the interval between onsets as well as to the
    walk's own tolerance, so that a truncated unit opening a phrase is not
    merged with the element before it.
    """
    reach = min(tolerance * period, interval / 2)
    while first > 0:
        earlier = np.arange(first)
        distances = np.abs(onsets[earlier] - (onsets[first] - period))
        if distances.min() > reach:
            break
        first = int(earlier[np.argmin(distances)])
    return first


def _first_unit(onsets, phase, period, tolerance):
    """Index of the first onset that falls on the phase grid."""
    for target in np.arange(phase, onsets[-1] + period, period):
        distances = np.abs(onsets - target)
        if distances.min() <= tolerance * period:
            return int(np.argmin(distances))
    return None
