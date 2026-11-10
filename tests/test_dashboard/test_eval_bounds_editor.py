# Licence: BSD-3-Clause
"""The bounds editor on the eval sample correction page."""
import types

import matplotlib
matplotlib.use("agg")
import matplotlib.figure
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from canapy.correction import make_annot_key
from dashboard.view.eval.samplecorrection import (
    SampleCorrectionDashboard, SampleCorrectorView, SingleSampleCorrectorView,
)

SR = 44100


@pytest.fixture()
def song(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1e-3, int(3.0 * SR))
    path = tmp_path / "song.wav"
    sf.write(str(path), y, SR)
    return str(path)


@pytest.fixture()
def card(song, config):
    from canapy.corpus import Corpus
    df = pd.DataFrame({
        "label": ["A", config.transforms.annots.silence_tag, "B"],
        "onset_s": [1.5, 1.8, 2.0],
        "offset_s": [1.8, 2.0, 2.4],
        "notated_path": [song] * 3,
    })
    corpus = Corpus.from_df(df, config=config)
    controler = types.SimpleNamespace(config=config, corpus=corpus)
    opened = []
    parent = types.SimpleNamespace(
        controler=controler,
        listen_correction=lambda label, increment: None,
        check_correction=lambda value: True,
        open_modal=opened.append,
        close_modal=lambda: opened.clear(),
        opened=opened,
    )
    # production passes a Series from iterrows(), not a namedtuple
    _, entry = next(df.iloc[:1].assign(**{"pred_syn-esn": "B"}).iterrows())
    spec = (matplotlib.figure.Figure(), np.zeros(10), np.zeros(10))
    return SingleSampleCorrectorView(entry, spec, parent, 1), controler


def test_the_card_knows_its_annotation_key(card, song):
    row, _ = card
    assert row.idx == make_annot_key(song, 1.5)


def test_nothing_is_pending_before_an_edit(card):
    row, _ = card
    assert row.bounds_correction is None
    assert row.bounds_status.object == ""


def test_the_editor_opens_in_the_modal(card):
    row, _ = card
    row._open_bounds_editor(None)
    assert len(row.parent.opened) == 1
    assert "bounds-modal" in row.parent.opened[0].css_classes


def test_validating_files_the_bounds_and_closes(card):
    row, _ = card
    row._open_bounds_editor(None)
    row._commit_bounds(row.idx, (1.42, 1.91))
    assert row.bounds_correction == pytest.approx((1.42, 1.91))
    assert "pending" in row.bounds_status.object
    assert row.parent.opened == []


def test_neighbours_come_from_the_corpus_not_the_misclassified_subset(card):
    # the misclassified segments hold only the problematic annotations, which
    # would place a boundary against an incomplete picture
    row, _ = card
    assert [n.label for n in row._neighbours()] == ["B"]


def test_silence_is_not_offered_as_a_neighbour(card, config):
    row, _ = card
    assert config.transforms.annots.silence_tag not in [
        n.label for n in row._neighbours()
    ]


def test_validating_redraws_the_card(card):
    from dashboard.controler.base import Controler
    row, controler = card
    for name in ("render_segment", "_segment_tick_reference"):
        setattr(controler, name, types.MethodType(getattr(Controler, name), controler))
    before = row.img.object

    row._commit_bounds(row.idx, (1.42, 1.91))

    assert row.img.object is not before


def test_both_audio_players_follow_the_new_bounds(card):
    import panel as pn
    from dashboard.controler.base import Controler
    row, controler = card
    for name in ("render_segment", "_segment_tick_reference"):
        setattr(controler, name, types.MethodType(getattr(Controler, name), controler))
    players = [p for p in row.audio_col if isinstance(p, pn.pane.Audio)]
    before = [p.object for p in players]

    row._commit_bounds(row.idx, (1.42, 1.91))

    assert all(p.object is not b for p, b in zip(players, before))
