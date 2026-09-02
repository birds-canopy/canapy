# Licence: BSD-3-Clause
import numpy as np
import pytest

from canapy.segmentation import group_segments, period_of, phase_of, segment_signal

SR = 44100
TONE_HZ = 4000.0
NOTE_S = 0.040
INTRA_S = 0.020
N_UNITS = 6


def _envelope(segments, duration, hop=0.001):
    """Rectangular energy envelope matching a list of segments."""
    times = np.arange(0, duration, hop)
    energy = np.full(len(times), 1e-6)
    for start, end in segments:
        energy[(times >= start) & (times < end)] = 1.0
    return times, energy


def _phrase(notes_per_unit, inter_s, gains=None, cut_notes=0):
    """Tone bursts laid out in units, and the ground truth at both levels."""
    rng = np.random.default_rng(0)
    gains = gains or (1.0,) * notes_per_unit
    chunks, units, notes, cursor = [], [], [], 0.05
    chunks.append(rng.normal(0, 1e-3, int(0.05 * SR)))

    for unit in range(N_UNITS):
        unit_start = cursor
        for index in range(notes_per_unit):
            if index:
                chunks.append(rng.normal(0, 1e-3, int(INTRA_S * SR)))
                cursor += INTRA_S
            n = int(NOTE_S * SR)
            t = np.arange(n) / SR
            chunks.append(gains[index] * np.sin(2 * np.pi * TONE_HZ * t)
                          + rng.normal(0, 1e-3, n))
            notes.append((cursor, cursor + NOTE_S))
            cursor += NOTE_S
        units.append((unit_start, cursor))
        if unit < N_UNITS - 1:
            chunks.append(rng.normal(0, 1e-3, int(inter_s * SR)))
            cursor += inter_s

    signal = np.concatenate(chunks)
    if cut_notes:
        offset = int((notes[cut_notes][0] - INTRA_S / 2) * SR)
        signal = signal[offset:]
        shift = offset / SR
        units = [(a - shift, b - shift) for a, b in units[1:]]
    return signal, units


def test_period_matches_the_repetition():
    period = 0.200
    segments = [(k * period, k * period + 0.05) for k in range(8)]
    times, energy = _envelope(segments, 8 * period)
    found, confidence = period_of(times, energy, min_period=0.05)
    assert found == pytest.approx(period, abs=0.005)
    assert confidence > 0.5


def test_long_silences_do_not_double_the_period():
    # a plain argmax locks onto 2T here, and merges the units two by two
    period = 0.150
    segments = [(k * period, k * period + 0.03) for k in range(10)]
    times, energy = _envelope(segments, 10 * period)
    found, _ = period_of(times, energy, min_period=0.03)
    assert found == pytest.approx(period, abs=0.005)


def test_phase_lands_where_the_cycle_starts():
    # not merely outside the loud part: on the step itself
    period = 0.200
    segments = [(k * period + 0.05, k * period + 0.15) for k in range(8)]
    times, energy = _envelope(segments, 8 * period)
    onsets = [start for start, _ in segments]
    phase = phase_of(onsets, times, energy, period)
    assert phase % period == pytest.approx(0.05, abs=0.02)


def test_phase_reads_amplitude_when_the_spacing_is_even():
    # evenly spaced notes, the unit marked only by a loud opening one
    signal, units = _phrase(notes_per_unit=3, inter_s=INTRA_S,
                            gains=(1.0, 0.45, 0.45))
    segments, _ = segment_signal(signal, SR, group_syllables=True)
    for (start, _), (true_start, _) in zip(segments, units):
        assert start == pytest.approx(true_start, abs=0.010)


def test_regular_elements_are_left_alone():
    segments = [(k * 0.1, k * 0.1 + 0.05) for k in range(8)]
    times, energy = _envelope(segments, 0.8)
    units, info = group_segments(segments, times, energy)
    assert not info["grouped"]
    assert units == segments


def test_two_notes_per_unit_are_merged():
    signal, units = _phrase(notes_per_unit=2, inter_s=0.060)
    segments, _ = segment_signal(signal, SR, group_syllables=True)
    assert len(segments) == len(units)
    for (start, _), (true_start, _) in zip(segments, units):
        assert start == pytest.approx(true_start, abs=0.010)


def test_grouping_is_opt_in():
    signal, units = _phrase(notes_per_unit=2, inter_s=0.060)
    segments, _ = segment_signal(signal, SR)
    assert len(segments) == 2 * len(units)


def test_a_phrase_starting_mid_unit_keeps_its_boundaries():
    # anchoring on the first detected element would shift every boundary by a
    # fraction of a cycle. The truncated opening unit is kept as its own
    # segment rather than dropped, so no annotated audio goes missing.
    signal, units = _phrase(notes_per_unit=3, inter_s=0.060, cut_notes=1)
    segments, _ = segment_signal(signal, SR, group_syllables=True)

    assert len(segments) == len(units) + 1
    for (start, _), (true_start, _) in zip(segments[1:], units):
        assert start == pytest.approx(true_start, abs=0.010)


def test_a_phrase_without_repetition_becomes_one_syllable():
    # one long syllable the detector cut into unevenly spaced pieces: no short
    # period fits, so the phrase holds barely one cycle
    segments = [(0.02, 0.07), (0.09, 0.115), (0.145, 0.19)]
    times, energy = _envelope(segments, 0.21)

    units, info = group_segments(segments, times, energy)
    assert info["single"]
    assert info["cycles"] < 1.6
    assert units == [(0.02, 0.19)]


def test_a_repeated_phrase_is_not_collapsed():
    segments = [(0.02 + k * 0.1, 0.07 + k * 0.1) for k in range(8)]
    times, energy = _envelope(segments, 0.82)

    units, info = group_segments(segments, times, energy)
    assert not info["single"]
    assert units == segments


def test_pooling_a_label_resists_one_bad_estimate():
    from canapy.segmentation.grouping import pool_by_label

    good = [{"period": 0.100, "single": False} for _ in range(9)]
    octave = [{"period": 0.200, "single": True}]
    pooled = pool_by_label({"A": good + octave})["A"]

    assert pooled["period"] == pytest.approx(0.100)
    assert pooled["n_phrases"] == 10


def test_pooling_does_not_decide_whether_a_phrase_repeats():
    # holding fewer than MIN_CYCLES periods divides the phrase's own duration,
    # so it is answered per phrase, not voted on
    from canapy.segmentation.grouping import pool_by_label

    rows = [{"period": 0.200, "single": True}] + [
        {"period": 0.100, "single": False} for _ in range(4)]
    pooled = pool_by_label({"A": rows})["A"]

    assert "single" not in pooled


def test_an_imposed_period_overrides_the_phrase():
    signal, units = _phrase(notes_per_unit=2, inter_s=0.060)
    on_its_own, _ = segment_signal(signal, SR, group_syllables=True)

    # twice the period merges the units two by two
    doubled, _ = segment_signal(signal, SR, group_syllables=True,
                                period=2 * (NOTE_S * 2 + INTRA_S + 0.060))
    assert len(on_its_own) == len(units)
    assert len(doubled) < len(on_its_own)


def test_an_imposed_single_verdict_collapses_the_phrase():
    signal, _ = _phrase(notes_per_unit=2, inter_s=0.060)
    segments, _ = segment_signal(signal, SR, group_syllables=True, single=True)
    assert len(segments) == 1
