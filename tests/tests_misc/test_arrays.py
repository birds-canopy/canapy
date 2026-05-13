# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
"""Tests for canapy/utils/arrays.py — _StructuredContainer, to_structured, total_size."""

import numpy as np
import pytest

from canapy.utils.arrays import _StructuredContainer, to_structured, total_size


class TestTotalSize:
    def test_single_array(self):
        arr = np.zeros((10, 5), dtype=np.float64)
        assert total_size([arr]) == arr.nbytes

    def test_multiple_arrays(self):
        a = np.zeros((10,), dtype=np.float32)
        b = np.ones((20,), dtype=np.int16)
        assert total_size([a, b]) == a.nbytes + b.nbytes

    def test_empty_list(self):
        assert total_size([]) == 0

    def test_dtype_matters(self):
        a32 = np.zeros(10, dtype=np.float32)
        a64 = np.zeros(10, dtype=np.float64)
        assert total_size([a64]) == 2 * total_size([a32])


class TestStructuredContainer:
    def _make(self):
        a = np.arange(10, dtype=np.float32).reshape(2, 5)
        b = np.arange(6, dtype=np.int16).reshape(2, 3)
        return _StructuredContainer(a=a, b=b), a, b

    def test_keys_present(self):
        c, _, _ = self._make()
        assert set(c.keys()) == {"a", "b"}

    def test_getitem_shape_preserved(self):
        c, a, b = self._make()
        assert c["a"].shape == a.shape
        assert c["b"].shape == b.shape

    def test_getitem_values_correct(self):
        c, a, b = self._make()
        np.testing.assert_array_equal(c["a"], a)
        np.testing.assert_array_equal(c["b"], b)

    def test_contains_existing_key(self):
        c, _, _ = self._make()
        assert "a" in c
        assert "b" in c

    def test_not_contains_missing_key(self):
        c, _, _ = self._make()
        assert "z" not in c

    def test_setitem_raises_not_implemented(self):
        c, a, _ = self._make()
        with pytest.raises(NotImplementedError):
            c["a"] = np.zeros((2, 5))

    def test_items_yields_all_pairs(self):
        c, a, b = self._make()
        result = dict(c.items())
        assert set(result.keys()) == {"a", "b"}
        np.testing.assert_array_equal(result["a"], a)
        np.testing.assert_array_equal(result["b"], b)

    def test_values_yields_correct_count(self):
        c, _, _ = self._make()
        assert len(list(c.values())) == 2

    def test_single_1d_array(self):
        arr = np.arange(20, dtype=np.float64)
        c = _StructuredContainer(x=arr)
        np.testing.assert_array_equal(c["x"], arr)


class TestToStructured:
    def test_returns_structured_container(self):
        a = np.zeros((3, 4), dtype=np.float64)
        result = to_structured({"a": a})
        assert isinstance(result, _StructuredContainer)

    def test_data_preserved(self):
        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        result = to_structured({"a": a})
        np.testing.assert_array_equal(result["a"], a)

    def test_multiple_fields(self):
        arrays = {
            "x": np.ones((5,), dtype=np.float32),
            "y": np.zeros((5, 2), dtype=np.int16),
        }
        result = to_structured(arrays)
        assert "x" in result
        assert "y" in result
