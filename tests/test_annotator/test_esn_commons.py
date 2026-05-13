# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
"""Tests for canapy/annotator/commons/esn.py — init_esn_model, maximum_a_posteriori."""

import numpy as np
import pytest

from canapy.annotator.commons.esn import init_esn_model, maximum_a_posteriori
from config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _esn_model_config(n_units=50):
    return Config(
        units=n_units,
        sr=0.4,
        leak=0.1,
        iss=0.0005,
        isd=0.02,
        isd2=0.002,
        ridge=1e-8,
        backend="threading",
        workers=1,
    )


# ---------------------------------------------------------------------------
# maximum_a_posteriori
# ---------------------------------------------------------------------------

class TestMaximumAPosteriori:
    def test_single_timestep_with_classes(self):
        logits = np.array([[0.1, 0.8, 0.1]])
        result = maximum_a_posteriori(logits, classes=["A", "B", "C"])
        assert list(result) == ["B"]

    def test_multiple_timesteps_with_classes(self):
        logits = np.array([
            [0.9, 0.1, 0.0],
            [0.0, 0.1, 0.9],
        ])
        result = maximum_a_posteriori(logits, classes=["X", "Y", "Z"])
        assert list(result) == ["X", "Z"]

    def test_without_classes_returns_integer_indices(self):
        logits = np.array([[0.2, 0.5, 0.3]])
        result = maximum_a_posteriori(logits)
        assert result[0] == 1

    def test_output_length_matches_n_timesteps(self):
        n = 30
        logits = np.random.default_rng(0).random((n, 5))
        classes = list("ABCDE")
        result = maximum_a_posteriori(logits, classes=classes)
        assert len(result) == n

    def test_all_results_come_from_classes(self):
        classes = list("ABC")
        logits = np.random.default_rng(0).random((20, 3))
        result = maximum_a_posteriori(logits, classes=classes)
        for r in result:
            assert r in classes

    def test_1d_input_treated_as_single_timestep(self):
        logits = np.array([0.1, 0.8, 0.1])
        result = maximum_a_posteriori(logits, classes=["A", "B", "C"])
        assert list(result) == ["B"]


# ---------------------------------------------------------------------------
# init_esn_model
# ---------------------------------------------------------------------------

class TestInitEsnModel:
    def test_returns_esn_object(self):
        from reservoirpy import ESN
        cfg = _esn_model_config()
        model = init_esn_model(cfg, input_dim=13, audio_features=["mfcc"], seed=42)
        assert isinstance(model, ESN)

    def test_model_has_reservoir(self):
        cfg = _esn_model_config()
        model = init_esn_model(cfg, input_dim=13, audio_features=["mfcc"], seed=0)
        assert hasattr(model, "reservoir")

    def test_model_has_readout(self):
        cfg = _esn_model_config()
        model = init_esn_model(cfg, input_dim=13, audio_features=["mfcc"], seed=0)
        assert hasattr(model, "readout")

    def _fit_tiny(self, model, n_in, n_cls=2, seed=0):
        """Fit model on minimal random data so readout is initialized."""
        rng = np.random.default_rng(seed)
        X = [rng.random((20, n_in)).astype(np.float64)]
        Y = [np.eye(n_cls)[rng.integers(0, n_cls, 20)].astype(np.float64)]
        model.fit(X, Y)
        return model

    def test_mfcc_only_runs_forward_pass_after_fit(self):
        cfg = _esn_model_config(n_units=30)
        model = init_esn_model(cfg, input_dim=13, audio_features=["mfcc"], seed=1)
        self._fit_tiny(model, n_in=13, n_cls=2)
        x = np.random.default_rng(1).random((10, 13)).astype(np.float64)
        out = model.run([x])
        assert out is not None

    def test_mfcc_delta_delta2_features_after_fit(self):
        cfg = _esn_model_config(n_units=30)
        model = init_esn_model(
            cfg, input_dim=13, audio_features=["mfcc", "delta", "delta2"], seed=2
        )
        self._fit_tiny(model, n_in=39, n_cls=2)
        x = np.random.default_rng(2).random((10, 39)).astype(np.float64)
        out = model.run([x])
        assert out is not None

    def _to_dense(self, mat):
        if hasattr(mat, "toarray"):
            return mat.toarray().astype(np.float64)
        return np.array(mat, dtype=np.float64)

    def test_different_seeds_produce_different_reservoir_weights(self):
        """Different seeds must produce models with different Win matrices."""
        cfg = _esn_model_config(n_units=50)
        m1 = init_esn_model(cfg, input_dim=13, audio_features=["mfcc"], seed=0)
        m2 = init_esn_model(cfg, input_dim=13, audio_features=["mfcc"], seed=999)
        # fit to materialise lazy sparse matrices
        self._fit_tiny(m1, n_in=13)
        self._fit_tiny(m2, n_in=13)
        w1 = self._to_dense(m1.reservoir.Win)
        w2 = self._to_dense(m2.reservoir.Win)
        assert not np.allclose(w1, w2), "Different seeds should produce different Win matrices"

    def test_same_seed_produces_identical_reservoir_weights(self):
        cfg = _esn_model_config(n_units=50)
        m1 = init_esn_model(cfg, input_dim=13, audio_features=["mfcc"], seed=42)
        m2 = init_esn_model(cfg, input_dim=13, audio_features=["mfcc"], seed=42)
        self._fit_tiny(m1, n_in=13)
        self._fit_tiny(m2, n_in=13)
        np.testing.assert_allclose(self._to_dense(m1.reservoir.Win), self._to_dense(m2.reservoir.Win), rtol=1e-6)
        np.testing.assert_allclose(self._to_dense(m1.reservoir.W), self._to_dense(m2.reservoir.W), rtol=1e-6)
