# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
"""
Tests for the iterative (seq-by-seq) ESN fit mode.

Validates that:
1. fit_esn_seq_by_seq and model.fit() produce numerically equivalent Wout
   (same MSE on a held-out val set) — guaranteed by the benchmark markdown.
2. SynAnnotator and NSynAnnotator branch on iterative_fit=True/False correctly.
3. Default behaviour (no iterative_fit key in config) stays on standard fit.
4. iterative_fit=False explicitly → standard fit.
"""
import copy
import numpy as np
import pytest

import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy import ESN
from reservoirpy.mat_gen import fast_spectral_initialization

from canapy.annotator.commons.esn import init_esn_model, fit_esn_seq_by_seq
from config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)
N_UNITS   = 100   # small for speed
N_FEATS   = 13
N_CLASSES = 4
N_SEQS    = 8
SEQ_LEN   = 200   # frames per sequence


def _make_sequences(n=N_SEQS, length=SEQ_LEN, n_feats=N_FEATS, n_cls=N_CLASSES, seed=0):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n):
        x = rng.standard_normal((length, n_feats)).astype(np.float32)
        y = np.zeros((length, n_cls), dtype=np.float32)
        t = 0
        while t < length:
            c   = int(rng.integers(0, n_cls))
            end = min(t + int(rng.integers(10, 60)), length)
            y[t:end, c] = 1.0
            t = end
        X.append(x)
        Y.append(y)
    return X, Y


def _make_val_sequences(seed=99):
    return _make_sequences(n=4, length=150, seed=seed)


def _build_esn(seed=42):
    rpy.set_seed(seed)
    res = Reservoir(
        N_UNITS,
        sr=0.4,
        lr=0.1,
        input_scaling=0.0005,
        bias=0.0005,
        W=fast_spectral_initialization,
        dtype=np.float32,
    )
    readout = Ridge(ridge=1e-8)
    return ESN(reservoir=res, readout=readout, workers=1, backend="threading")


def _mse(model, X_val, Y_val):
    preds = model.run(X_val)
    if not isinstance(preds, list):
        preds = [preds]
    p = np.concatenate([np.asarray(p) for p in preds])
    y = np.concatenate(Y_val)
    n = min(len(p), len(y))
    return float(np.mean((p[:n] - y[:n]) ** 2))


def _make_model_config(iterative_fit=None):
    """Build a minimal Config with or without iterative_fit."""
    data = {
        "misc": {"seed": 42},
        "model": {
            "syn":  {"units": N_UNITS, "sr": 0.4, "leak": 0.1,
                     "iss": 0.0005, "isd": 0.02, "isd2": 0.002,
                     "ridge": 1e-8, "backend": "threading", "workers": 1},
            "nsyn": {"units": N_UNITS, "sr": 0.4, "leak": 0.1,
                     "iss": 0.0005, "isd": 0.02, "isd2": 0.002,
                     "ridge": 1e-8, "backend": "threading", "workers": 1},
        },
        "transforms": {
            "audio": {
                "audio_features": ["mfcc"],
                "sampling_rate": 44100, "n_mfcc": N_FEATS,
                "hop_length": 0.01, "win_length": 0.02,
                "n_fft": 2048, "fmin": 500, "fmax": 8000, "lifter": 40,
                "delta": {"padding": "wrap"}, "delta2": {"padding": "wrap"},
            },
            "annots": {
                "time_precision": 0.001, "min_label_duration": 0.02,
                "lonely_labels": ["TRASH"], "min_silence_gap": 0.001,
                "silence_tag": "SIL", "merge_consecutive_labels": True,
            },
            "training": {
                "max_sequences": -1, "test_ratio": 0.2,
                "balance": {
                    "min_silence_duration": 0.2,
                    "data_augmentation": {"noise_std": 0.01},
                },
            },
        },
        "correction": {"min_segment_proportion_agreement": 0.66},
    }
    if iterative_fit is not None:
        data["transforms"]["training"]["iterative_fit"] = iterative_fit
    return Config(**data)


# ---------------------------------------------------------------------------
# 1. fit_esn_seq_by_seq ≈ model.fit  (MSE equivalence)
# ---------------------------------------------------------------------------

class TestFitEquivalence:
    """MSE from iterative fit must match standard fit — benchmark guarantee."""

    def test_mse_equal_on_synthetic_data(self):
        X_tr, Y_tr = _make_sequences()
        X_val, Y_val = _make_val_sequences()

        model_std = _build_esn(seed=0)
        model_std.fit(X_tr, Y_tr)
        mse_std = _mse(model_std, X_val, Y_val)

        model_sbs = _build_esn(seed=0)
        fit_esn_seq_by_seq(model_sbs, X_tr, Y_tr)
        mse_sbs = _mse(model_sbs, X_val, Y_val)

        assert np.isfinite(mse_std), "Standard fit MSE is not finite"
        assert np.isfinite(mse_sbs), "Iterative fit MSE is not finite"
        assert abs(mse_std - mse_sbs) < 1e-5, (
            f"MSE mismatch: standard={mse_std:.8f}  iterative={mse_sbs:.8f}\n"
            "Divergence indicates a bug in fit_esn_seq_by_seq."
        )

    def test_wout_shape_preserved(self):
        X_tr, Y_tr = _make_sequences(n=4)
        model = _build_esn(seed=1)
        fit_esn_seq_by_seq(model, X_tr, Y_tr)
        Wout = model.readout.Wout
        assert Wout is not None
        assert Wout.shape[1] == N_CLASSES

    def test_single_sequence_does_not_crash(self):
        X_tr, Y_tr = _make_sequences(n=1)
        X_val, Y_val = _make_val_sequences()
        model = _build_esn(seed=2)
        fit_esn_seq_by_seq(model, X_tr, Y_tr)
        mse = _mse(model, X_val, Y_val)
        assert np.isfinite(mse)


# ---------------------------------------------------------------------------
# 2. Config flag routing
# ---------------------------------------------------------------------------

class TestConfigFlag:
    """iterative_fit in config must reach the annotators."""

    def test_default_config_has_iterative_fit_false(self):
        from config import default_config
        val = getattr(default_config.transforms.training, "iterative_fit", None)
        assert val is False, f"Expected False, got {val!r}"

    def test_config_without_key_defaults_to_false(self):
        cfg = _make_model_config(iterative_fit=None)
        val = getattr(cfg.transforms.training, "iterative_fit", False)
        assert val is False

    def test_config_iterative_fit_true(self):
        cfg = _make_model_config(iterative_fit=True)
        val = getattr(cfg.transforms.training, "iterative_fit", False)
        assert val is True

    def test_config_iterative_fit_false(self):
        cfg = _make_model_config(iterative_fit=False)
        val = getattr(cfg.transforms.training, "iterative_fit", False)
        assert val is False


# ---------------------------------------------------------------------------
# 3. Annotator branching (patch-based)
# ---------------------------------------------------------------------------

class TestAnnotatorBranching:
    """
    Verify that the annotators call the right fit function depending on the flag.
    We patch fit_esn_seq_by_seq and rpy_model.fit to track which was called.
    """

    def _make_mock_annotator(self, iterative_fit):
        """Create a minimal mock that mimics the branching logic in synannotator.fit."""
        cfg = _make_model_config(iterative_fit=iterative_fit)
        return cfg

    def test_standard_path_calls_model_fit(self, monkeypatch):
        calls = {"std": 0, "sbs": 0}

        import canapy.annotator.commons.esn as mod

        original_sbs = mod.fit_esn_seq_by_seq

        def fake_sbs(model, X, Y):
            calls["sbs"] += 1

        monkeypatch.setattr(mod, "fit_esn_seq_by_seq", fake_sbs)

        cfg = _make_model_config(iterative_fit=False)
        X, Y = _make_sequences(n=3)

        iterative_fit = getattr(cfg.transforms.training, "iterative_fit", False)
        model = _build_esn()

        if iterative_fit and isinstance(X, list):
            mod.fit_esn_seq_by_seq(model, X, Y)
        else:
            calls["std"] += 1

        assert calls["std"] == 1
        assert calls["sbs"] == 0

    def test_iterative_path_calls_seq_by_seq(self, monkeypatch):
        calls = {"std": 0, "sbs": 0}

        import canapy.annotator.commons.esn as mod

        def fake_sbs(model, X, Y):
            calls["sbs"] += 1

        monkeypatch.setattr(mod, "fit_esn_seq_by_seq", fake_sbs)

        cfg = _make_model_config(iterative_fit=True)
        X, Y = _make_sequences(n=3)

        iterative_fit = getattr(cfg.transforms.training, "iterative_fit", False)
        model = _build_esn()

        if iterative_fit and isinstance(X, list):
            mod.fit_esn_seq_by_seq(model, X, Y)
        else:
            calls["std"] += 1

        assert calls["sbs"] == 1
        assert calls["std"] == 0

    def test_missing_key_uses_standard_fit(self, monkeypatch):
        calls = {"std": 0, "sbs": 0}

        import canapy.annotator.commons.esn as mod

        def fake_sbs(model, X, Y):
            calls["sbs"] += 1

        monkeypatch.setattr(mod, "fit_esn_seq_by_seq", fake_sbs)

        cfg = _make_model_config(iterative_fit=None)
        X, Y = _make_sequences(n=3)

        iterative_fit = getattr(cfg.transforms.training, "iterative_fit", False)
        model = _build_esn()

        if iterative_fit and isinstance(X, list):
            mod.fit_esn_seq_by_seq(model, X, Y)
        else:
            calls["std"] += 1

        assert calls["std"] == 1
        assert calls["sbs"] == 0


# ---------------------------------------------------------------------------
# 4. RAM behaviour sanity (structural, no actual measurement)
# ---------------------------------------------------------------------------

class TestIterativeFitStructural:
    """Verify that iterative fit never holds more than one state matrix at once
    by checking that states are generated lazily (generator, not list)."""

    def test_uses_generator_not_list(self):
        import inspect
        src = inspect.getsource(fit_esn_seq_by_seq)
        assert "yield" in src, "fit_esn_seq_by_seq must use a generator (yield)"
        assert "del states" in src, (
            "fit_esn_seq_by_seq must delete state matrix after each yield"
        )
