# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
"""Tests for canapy/annotator/commons/compat.py — _Compat and _CompatModelUnpickler."""

import io
import pickle
from types import SimpleNamespace

import pytest

from canapy.annotator.commons.compat import _Compat, _CompatModelUnpickler


class TestCompat:
    def test_is_subclass_of_simple_namespace(self):
        assert issubclass(_Compat, SimpleNamespace)

    def test_instantiates_empty(self):
        obj = _Compat()
        assert isinstance(obj, _Compat)

    def test_keyword_args_become_attributes(self):
        obj = _Compat(esn="model_obj", vocab=["A", "B", "SIL"])
        assert obj.esn == "model_obj"
        assert obj.vocab == ["A", "B", "SIL"]

    def test_arbitrary_attributes_accessible(self):
        obj = _Compat(foo=42, bar=3.14, baz=True)
        assert obj.foo == 42
        assert obj.bar == pytest.approx(3.14)
        assert obj.baz is True

    def test_attribute_can_be_set_after_init(self):
        obj = _Compat()
        obj.new_attr = "hello"
        assert obj.new_attr == "hello"

    def test_equality_of_two_compat_instances(self):
        a = _Compat(x=1, y=2)
        b = _Compat(x=1, y=2)
        assert a == b


class TestCompatModelUnpickler:
    def test_loads_plain_dict(self):
        data = {"key": "value", "num": 42}
        buf = io.BytesIO(pickle.dumps(data))
        result = _CompatModelUnpickler(buf).load()
        assert result == data

    def test_loads_list(self):
        data = [1, 2, 3, "abc"]
        buf = io.BytesIO(pickle.dumps(data))
        result = _CompatModelUnpickler(buf).load()
        assert result == data

    def test_loads_simple_namespace(self):
        ns = SimpleNamespace(a=1, b="two")
        buf = io.BytesIO(pickle.dumps(ns))
        result = _CompatModelUnpickler(buf).load()
        assert result.a == 1
        assert result.b == "two"

    def test_loads_compat_object(self):
        obj = _Compat(esn="fake_esn", vocab=["X"])
        buf = io.BytesIO(pickle.dumps(obj))
        result = _CompatModelUnpickler(buf).load()
        assert result.vocab == ["X"]

    def test_loads_numpy_array(self):
        import numpy as np
        arr = np.arange(10)
        buf = io.BytesIO(pickle.dumps(arr))
        result = _CompatModelUnpickler(buf).load()
        np.testing.assert_array_equal(result, arr)
