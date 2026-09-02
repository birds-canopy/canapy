# Licence: BSD-3-Clause
"""Which annotators the Annotate page offers, and where it finds them."""
import types
from pathlib import Path

import pytest

from dashboard.controler.base import Controler
from dashboard.view.annotate.annotate_dash import AnnotateDashboard


def _scan(model_root):
    owner = types.SimpleNamespace(controler=types.SimpleNamespace(model_root=model_root))
    return AnnotateDashboard._scan_models(owner)


def _exported(tmp_path, *names):
    export_dir = tmp_path / "exported_models"
    export_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (export_dir / name).write_bytes(b"pickle")
    return tmp_path


def test_both_sub_models_are_an_ensemble(tmp_path):
    root = _exported(tmp_path, "syn-esn", "nsyn-esn")
    found = _scan(root)
    assert set(found) == {"syn-esn", "nsyn-esn", "ensemble"}


def test_one_sub_model_is_not_an_ensemble(tmp_path):
    found = _scan(_exported(tmp_path, "syn-esn"))
    assert set(found) == {"syn-esn"}


def test_an_exported_ensemble_keeps_its_own_path(tmp_path):
    root = _exported(tmp_path, "syn-esn", "nsyn-esn", "ensemble")
    found = _scan(root)
    assert found["ensemble"].endswith("ensemble")


def test_nothing_exported_yet(tmp_path):
    assert _scan(tmp_path) == {}
    assert _scan(None) == {}


@pytest.mark.parametrize("name", ["syn-esn", "ensemble"])
def test_a_model_is_found_inside_the_export_directory(tmp_path, name):
    root = _exported(tmp_path, name)
    controler = types.SimpleNamespace(model_root=root, output_directory=tmp_path)
    found = Controler._find_model_on_disk(controler, name)
    assert found == root / "exported_models" / name


def test_a_model_is_found_straight_under_the_root(tmp_path):
    (tmp_path / "syn-esn").write_bytes(b"pickle")
    controler = types.SimpleNamespace(model_root=tmp_path, output_directory=tmp_path)
    assert Controler._find_model_on_disk(controler, "syn-esn") == tmp_path / "syn-esn"


def test_a_missing_model_is_reported_as_missing(tmp_path):
    controler = types.SimpleNamespace(model_root=tmp_path, output_directory=tmp_path)
    assert Controler._find_model_on_disk(controler, "nsyn-esn") is None
