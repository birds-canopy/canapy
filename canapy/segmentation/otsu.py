# Licence: BSD-3-Clause
"""Otsu's threshold, and the quality measure that says when not to trust it."""
import numpy as np

N_BINS = 256


def otsu(values, n_bins=N_BINS):
    """Threshold of a bimodal distribution, and how bimodal it actually was.

    Returns ``(threshold, eta_squared)``. Otsu always returns a threshold, even
    on a unimodal distribution where it means nothing; ``eta_squared`` is the
    share of total variance the split explains, and is the only signal that
    the split was justified.
    """
    hist, edges = np.histogram(values, bins=n_bins)
    centres = (edges[:-1] + edges[1:]) / 2

    counts = np.cumsum(hist)
    above = counts[-1] - counts
    cumulative = np.cumsum(hist * centres)
    mean_below = cumulative / np.maximum(counts, 1)
    mean_above = (cumulative[-1] - cumulative) / np.maximum(above, 1)

    # counts rather than proportions scales the criterion by N**2, which the
    # argmax does not see; the division is put back in when scaling eta squared
    criterion = (counts * above * (mean_below - mean_above) ** 2)[:-1]
    best = int(np.argmax(criterion))

    total = counts[-1]
    mean = cumulative[-1] / max(total, 1)
    variance = (hist * (centres - mean) ** 2).sum() / max(total, 1)

    eta_squared = (
        float(criterion[best] / total**2 / variance) if total and variance > 0 else 0.0
    )
    return float(centres[best]), eta_squared


def otsu_threshold(values, n_bins=N_BINS):
    return otsu(values, n_bins=n_bins)[0]
