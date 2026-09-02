# Licence: BSD-3-Clause
import numpy as np
import pytest

from canapy.segmentation import mask_to_segments, segment_signal
from canapy.segmentation.features import NPERSEG

SR = 44100
TONE_HZ = 4000.0
BURST_S = 0.040
GAP_S = 0.020
N_BURSTS = 3


def _mask(pattern, hop=0.001):
    mask = np.array(pattern, dtype=bool)
    return mask, np.arange(len(mask)) * hop


@pytest.fixture()
def bursts():
    """Three tone bursts separated by silence, and their true bounds."""
    rng = np.random.default_rng(0)
    signal, bounds, cursor = [], [], 0.0

    for i in range(N_BURSTS):
        if i:
            signal.append(rng.normal(0, 1e-3, int(GAP_S * SR)))
            cursor += GAP_S
        n = int(BURST_S * SR)
        t = np.arange(n) / SR
        signal.append(np.sin(2 * np.pi * TONE_HZ * t) + rng.normal(0, 1e-3, n))
        bounds.append((cursor, cursor + BURST_S))
        cursor += BURST_S

    return np.concatenate(signal), bounds


def test_contiguous_runs_become_segments():
    mask, times = _mask([0, 1, 1, 1, 0, 0, 1, 1, 1, 0])
    segments = mask_to_segments(mask, times, min_syllable_duration=0.0,
                                min_silence_gap=0.0)
    assert len(segments) == 2


def test_merging_happens_before_filtering():
    # two halves of one syllable, 2 ms each, split by a 2 ms gap. Filtering
    # first would discard both for being under min_syllable_duration; merging
    # first glues them into one 6 ms syllable that survives
    mask, times = _mask([1, 1, 1, 0, 1, 1, 1])
    segments = mask_to_segments(mask, times, min_syllable_duration=0.004,
                                min_silence_gap=0.003)
    assert len(segments) == 1
    assert segments[0] == pytest.approx((0.0, 0.006))


def test_gap_wider_than_min_silence_gap_is_kept():
    mask, times = _mask([1, 1, 0, 0, 0, 1, 1])
    segments = mask_to_segments(mask, times, min_syllable_duration=0.0,
                                min_silence_gap=0.002)
    assert len(segments) == 2


def test_short_segments_are_discarded():
    mask, times = _mask([1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    segments = mask_to_segments(mask, times, min_syllable_duration=0.004,
                                min_silence_gap=0.001)
    assert len(segments) == 1


def test_signal_shorter_than_one_window_is_refused():
    segments, eta_squared = segment_signal(np.zeros(NPERSEG - 1), SR)
    assert segments == []
    assert eta_squared == 0.0


def test_empty_signal_is_refused():
    assert segment_signal(np.zeros(0), SR) == ([], 0.0)


def test_bursts_are_recovered(bursts):
    # bounds are spread by half an analysis window (5.8 ms at this rate)
    signal, expected = bursts
    segments, eta_squared = segment_signal(signal, SR)

    assert len(segments) == N_BURSTS
    assert eta_squared > 0.5
    for (start, end), (true_start, true_end) in zip(segments, expected):
        assert start == pytest.approx(true_start, abs=0.010)
        assert end == pytest.approx(true_end, abs=0.010)


def test_flat_gate_can_only_remove_segments(bursts):
    signal, _ = bursts
    wide, _ = segment_signal(signal, SR, wiener_max=0.0)
    narrow, _ = segment_signal(signal, SR, wiener_max=-3.0)
    assert len(narrow) <= len(wide)


def test_eta2_floor_refuses_the_split(bursts):
    signal, _ = bursts
    segments, eta_squared = segment_signal(signal, SR, eta2_min=1.0)
    assert segments == []
    assert eta_squared < 1.0
