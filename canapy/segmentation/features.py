# Licence: BSD-3-Clause
"""Signal contours feeding the syllable segmenter.

Unrelated to ``transforms.audio``: a 0.5 ms hop, a raw power STFT, the file's
native sampling rate and a narrower band. Computed on the fly, never cached.
"""
import numpy as np
from scipy.signal import stft

NPERSEG = 512
SMOOTH_S = 0.003


def contours(signal, sr, band, hop, nperseg=NPERSEG, smooth=SMOOTH_S):
    """Frame times, band energy, the same in dB below peak, and flatness.

    All of it comes from a single STFT, ~97% overlapped at this hop. The
    grouping autocorrelates the linear energy; thresholding wants the dB one.
    """
    hop_samples = max(1, int(hop * sr))
    freqs, times, spectrum = stft(
        signal, fs=sr, nperseg=nperseg, noverlap=nperseg - hop_samples
    )
    power = np.abs(spectrum) ** 2
    energy = _band_energy(freqs, power, band, hop, smooth)

    return times, energy, _to_db(energy), _flatness(freqs, power, band)


def _band_energy(freqs, power, band, hop, smooth):
    keep = (freqs >= band[0]) & (freqs < band[1])
    energy = power[keep].sum(axis=0)

    width = max(1, int(smooth / hop))
    return np.convolve(energy, np.ones(width) / width, mode="same")


def _to_db(energy):
    # relative to the peak of this phrase, so a dB threshold carries over
    # from a loud phrase to a quiet one
    db = 10 * np.log10(energy + 1e-20)
    return db - db.max()


def _flatness(freqs, power, band):
    """Wiener entropy: log ratio of geometric to arithmetic mean, per frame."""
    keep = (freqs >= band[0]) & (freqs <= band[1])
    band_power = power[keep] + 1e-20
    return np.log10(np.exp(np.log(band_power).mean(0)) / band_power.mean(0))
