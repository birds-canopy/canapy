# Licence: BSD-3-Clause
"""The repertoire cache is bounded, and releases the figures it drops."""
import gc
import types
import weakref

import matplotlib
matplotlib.use("agg")
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import dashboard.controler.base as base
from dashboard.controler.base import Controler


@pytest.fixture()
def controler(monkeypatch, config):
    monkeypatch.setattr(base, "REPERTOIRE_CACHE_SAMPLES", 12)
    monkeypatch.setattr(
        base, "plot_segment_melspectrogram",
        # plots/base.py builds figures directly, not through pyplot, so a
        # dropped reference is what frees them
        lambda *a, **k: (matplotlib.figure.Figure(), [0.0], [0.0]),
    )
    df = pd.DataFrame({
        "label": ["A"] * 40,
        "onset_s": [i * 0.2 for i in range(40)],
        "offset_s": [i * 0.2 + 0.1 for i in range(40)],
        "notated_path": ["/song.wav"] * 40,
    })
    ctl = types.SimpleNamespace(
        config=config,
        corpus=types.SimpleNamespace(dataset=df.assign(label="A")),
        _repertoire_cache=base.OrderedDict(),
    )
    for name in ("load_repertoire", "_evict_repertoire_cache", "clear_repertoire_cache",
                 "_segment_tick_reference", "render_segment"):
        setattr(ctl, name, types.MethodType(getattr(Controler, name), ctl))
    return ctl, df


def _page(ctl, df, start, size=4):
    return ctl.load_repertoire(df.iloc[start:start + size])


def test_a_page_is_served_from_the_cache_the_second_time(controler):
    ctl, df = controler
    assert _page(ctl, df, 0) is _page(ctl, df, 0)


def test_paging_through_a_class_does_not_grow_without_bound(controler):
    ctl, df = controler
    for start in range(0, 40, 4):
        _page(ctl, df, start)
    held = sum(len(specs) for specs in ctl._repertoire_cache.values())
    assert held <= base.REPERTOIRE_CACHE_SAMPLES


def test_an_evicted_page_is_released(controler):
    ctl, df = controler
    first = _page(ctl, df, 0)
    dead = [weakref.ref(fig) for fig, _, _ in first]
    del first
    for start in range(4, 40, 4):
        _page(ctl, df, start)
    gc.collect()
    assert all(ref() is None for ref in dead)


def test_the_least_recently_used_page_goes_first(controler):
    ctl, df = controler
    kept = _page(ctl, df, 0)
    evicted = _page(ctl, df, 4)
    _page(ctl, df, 8)          # cache full: 3 pages of 4 for a cap of 12
    _page(ctl, df, 0)          # re-visited, so no longer the oldest
    _page(ctl, df, 12)         # pushes one page out — the one not touched since

    assert kept is ctl._repertoire_cache[("A", tuple(df.index[0:4]))]
    assert ("A", tuple(df.index[4:8])) not in ctl._repertoire_cache
    assert evicted is not None


def test_clearing_releases_every_figure(controler):
    ctl, df = controler
    dead = [weakref.ref(fig) for fig, _, _ in _page(ctl, df, 0)]
    ctl.clear_repertoire_cache()
    gc.collect()
    assert ctl._repertoire_cache == {}
    assert all(ref() is None for ref in dead)
