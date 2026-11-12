# Licence: BSD-3-Clause
"""Detection of mislabeled segments: their mean MFCC embedded by UMAP, the
segments whose label does not match their neighbourhood, and
the files whose annotations seem shifted in time.
"""
from .base import find_outliers, find_shifted_files, outliers_params

__all__ = [
    "find_outliers",
    "find_shifted_files",
    "outliers_params",
]
