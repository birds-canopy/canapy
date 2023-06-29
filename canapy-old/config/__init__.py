import os
import json


_DEFAULT_CONFIG = {
    "syn": {
        "N": 1000,
        "sr": 0.4,
        "leak": 1e-1,
        "iss": 5e-4,
        "isd": 2e-2,
        "isd2": 2e-3,
        "ridge": 1e-8,
    },
    "nsyn": {
        "N": 1000,
        "sr": 0.4,
        "leak": 1e-1,
        "iss": 5e-4,
        "isd": 2e-2,
        "isd2": 2e-3,
        "ridge": 1e-8,
    },
    "sampling_rate": 44100,  # canary
    # "sampling_rate": 32000,  # zebra finches
    "strict_window": True,
    "n_fft": 0.04,
    "hop_length": 0.01,
    "win_length": 0.02,
    "n_mfcc": 13,
    "lifter": 40,
    "fmin": 500,
    "fmax": 8000,
    "mfcc": True,
    "delta": True,
    "delta2": True,
    "padding": "wrap",
    "min_class_duration": 30,
    "min_silence_duration": 0.2,
    "min_analysis_window_per_sample": 2,
    "min_correct_timesteps_per_sample": 0.66,
    "min_frame_nb": 2,
    "keep_separate": ["cri"],
    "test_ratio": 0.2,
    "seed": None,
}


def default_config():
    return _DEFAULT_CONFIG
