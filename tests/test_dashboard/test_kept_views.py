# Licence: BSD-3-Clause
"""The Annotate page survives navigation until what it was built from changes."""
import os
import types

from dashboard.app import CanapyDashboard
from dashboard.view.annotate.annotate_dash import AnnotateDashboard


def _controler(tmp_path, **overrides):
    values = dict(
        config=types.SimpleNamespace(data={"transforms": {"sr": 44100},
                                           "segmentation": {"pool_period": False}}),
        model_root=tmp_path,
        audio_directory=tmp_path / "audio",
        output_directory=tmp_path / "out",
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _export(tmp_path, name, content=b"pickle"):
    export_dir = tmp_path / "exported_models"
    export_dir.mkdir(exist_ok=True)
    (export_dir / name).write_bytes(content)
    return export_dir / name


class _Page:
    built = 0

    def __init__(self, parent):
        type(self).built += 1
        self.refreshed = 0

    dependencies = staticmethod(lambda controler: controler.deps)

    def refresh(self):
        self.refreshed += 1


def _app(deps):
    _Page.built = 0
    return types.SimpleNamespace(
        views={"annotate": _Page, "home": lambda parent: object()},
        controler=types.SimpleNamespace(deps=deps),
        _kept_views={},
    )


def test_page_is_reused_while_dependencies_hold():
    app = _app("a")
    first = CanapyDashboard._view_for(app, "annotate")
    CanapyDashboard._view_for(app, "home")
    again = CanapyDashboard._view_for(app, "annotate")
    assert again is first
    assert _Page.built == 1
    assert first.refreshed == 1


def test_page_is_rebuilt_when_a_dependency_changes():
    app = _app("a")
    first = CanapyDashboard._view_for(app, "annotate")
    app.controler.deps = "b"
    assert CanapyDashboard._view_for(app, "annotate") is not first
    assert _Page.built == 2


def test_views_without_dependencies_are_always_rebuilt():
    app = _app("a")
    assert CanapyDashboard._view_for(app, "home") is not CanapyDashboard._view_for(app, "home")


def test_segmentation_settings_do_not_count(tmp_path):
    controler = _controler(tmp_path)
    before = AnnotateDashboard.dependencies(controler)
    controler.config.data["segmentation"]["pool_period"] = True
    assert AnnotateDashboard.dependencies(controler) == before


def test_other_config_sections_count(tmp_path):
    controler = _controler(tmp_path)
    before = AnnotateDashboard.dependencies(controler)
    controler.config.data["transforms"]["sr"] = 32000
    assert AnnotateDashboard.dependencies(controler) != before


def test_dataset_and_model_root_count(tmp_path):
    controler = _controler(tmp_path)
    before = AnnotateDashboard.dependencies(controler)
    assert AnnotateDashboard.dependencies(
        _controler(tmp_path, audio_directory=tmp_path / "other")) != before
    assert AnnotateDashboard.dependencies(
        _controler(tmp_path, model_root=tmp_path / "other")) != before


def test_a_retrained_model_counts(tmp_path):
    model = _export(tmp_path, "syn-esn")
    controler = _controler(tmp_path)
    before = AnnotateDashboard.dependencies(controler)
    stat = model.stat()
    os.utime(model, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert AnnotateDashboard.dependencies(controler) != before


def test_a_newly_exported_model_counts(tmp_path):
    _export(tmp_path, "syn-esn")
    controler = _controler(tmp_path)
    before = AnnotateDashboard.dependencies(controler)
    _export(tmp_path, "nsyn-esn")
    assert AnnotateDashboard.dependencies(controler) != before


def test_before_any_dataset_is_loaded(tmp_path):
    controler = types.SimpleNamespace(audio_directory=None, output_directory=tmp_path)
    AnnotateDashboard.dependencies(controler)
