# Licence: BSD-3-Clause
"""From the Edit bounds button to the corrected corpus."""
import types

import matplotlib
matplotlib.use("agg")
import matplotlib.figure
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import dashboard.controler.base as base
from dashboard.controler.base import Controler
from dashboard.view.preprocess.preprocess_dash import SingleSampleRow

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
        "label": ["A", "B"],
        "onset_s": [1.5, 2.0],
        "offset_s": [1.8, 2.4],
        "notated_path": [song] * 2,
    })
    corpus = Corpus.from_df(df, config=config)
    controler = types.SimpleNamespace(config=config, corpus=corpus)
    controler.compute_classes = lambda: None
    controler.apply_live_corrections = types.MethodType(
        Controler.apply_live_corrections, controler
    )
    controler.upload_corrections = types.MethodType(
        Controler.upload_corrections, controler
    )
    controler._drop_stale_training_resources = types.MethodType(
        Controler._drop_stale_training_resources, controler
    )
    controler._correction_store = {"class": {}, "annot": {}, "bounds": {}}
    controler._metrics_store = {"misclass": None}
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
    _, row = next(df.iloc[:1].iterrows())
    spec = (matplotlib.figure.Figure(), np.zeros(10), np.zeros(10))
    return SingleSampleRow(row, spec, parent, "k", 1), controler


def test_nothing_is_pending_before_an_edit(card):
    row, _ = card
    assert row.bounds_correction is None
    assert row.bounds_status.object == ""


def test_validating_files_the_new_bounds(card):
    row, _ = card
    row._commit_bounds("k", (1.42, 1.91))
    assert row.bounds_correction == pytest.approx((1.42, 1.91))
    assert "pending" in row.bounds_status.object


def test_the_editor_opens_in_the_modal_and_closes(card):
    row, _ = card
    row._open_bounds_editor(None)
    assert len(row.parent.opened) == 1
    assert "bounds-modal" in row.parent.opened[0].css_classes
    row._close_bounds_editor()
    assert row.parent.opened == []


def test_validating_closes_the_modal(card):
    row, _ = card
    row._open_bounds_editor(None)
    row._commit_bounds("k", (1.42, 1.91))
    assert row.parent.opened == []


def test_the_neighbour_of_the_same_file_is_offered(card):
    row, _ = card
    neighbours = row._neighbours()
    assert [n.label for n in neighbours] == ["B"]


def test_an_unresolvable_key_changes_nothing(card):
    row, controler = card
    before = controler.corpus.dataset.copy()

    controler.apply_live_corrections({"k": (1.42, 1.91)}, "bounds")

    after = controler.corpus.dataset
    assert list(after.onset_s) == list(before.onset_s)
    assert list(after.offset_s) == list(before.offset_s)


def test_a_real_key_moves_its_annotation(card):
    from canapy.correction import make_annot_key
    row, controler = card
    df = controler.corpus.dataset
    key = make_annot_key(df.iloc[0].notated_path, df.iloc[0].onset_s)

    controler.apply_live_corrections({key: (1.42, 1.91)}, "bounds")

    moved = controler.corpus.dataset
    moved = moved[moved.label == "A"].iloc[0]
    assert (moved.onset_s, moved.offset_s) == pytest.approx((1.42, 1.91))


def test_the_correction_is_filed_in_the_store(card):
    from canapy.correction import make_annot_key
    row, controler = card
    df = controler.corpus.dataset
    key = make_annot_key(df.iloc[0].notated_path, df.iloc[0].onset_s)

    controler.apply_live_corrections({key: (1.42, 1.91)}, "bounds")

    assert controler._correction_store["bounds"] == {key: (1.42, 1.91)}


def test_silence_is_not_offered_as_a_neighbour(card, song, config):
    # growing a phrase into the silence beside it is the normal correction
    from canapy.corpus import Corpus
    row, controler = card
    df = pd.DataFrame({
        "label": ["A", config.transforms.annots.silence_tag, "B"],
        "onset_s": [1.5, 1.8, 2.0],
        "offset_s": [1.8, 2.0, 2.4],
        "notated_path": [song] * 3,
    })
    controler.corpus = Corpus.from_df(df, config=config)

    assert [n.label for n in row._neighbours()] == ["B"]


def test_the_misclassified_copy_follows_the_corpus(card):
    from canapy.correction import make_annot_key
    row, controler = card
    df = controler.corpus.dataset
    controler._metrics_store = {"misclass": df.copy()}
    key = make_annot_key(df.iloc[0].notated_path, df.iloc[0].onset_s)

    controler.apply_live_corrections({key: (1.42, 1.91)}, "bounds")

    misclass = controler._metrics_store["misclass"]
    moved = misclass[misclass.label == "A"].iloc[0]
    assert (moved.onset_s, moved.offset_s) == pytest.approx((1.42, 1.91))


def test_a_neighbour_trimmed_in_the_corpus_is_trimmed_in_the_copy(card):
    from canapy.correction import make_annot_key
    row, controler = card
    df = controler.corpus.dataset
    controler._metrics_store = {"misclass": df.copy()}
    key = make_annot_key(df.iloc[0].notated_path, df.iloc[0].onset_s)

    controler.apply_live_corrections({key: (1.5, 2.2)}, "bounds")

    misclass = controler._metrics_store["misclass"]
    assert misclass[misclass.label == "B"].iloc[0].onset_s == pytest.approx(2.2)


def test_an_empty_misclassified_copy_is_left_alone(card):
    from canapy.correction import make_annot_key
    row, controler = card
    df = controler.corpus.dataset
    empty = df.iloc[:0]
    controler._metrics_store = {"misclass": empty}
    key = make_annot_key(df.iloc[0].notated_path, df.iloc[0].onset_s)

    controler.apply_live_corrections({key: (1.42, 1.91)}, "bounds")

    assert controler._metrics_store["misclass"] is empty


def test_validating_redraws_the_card(card):
    row, controler = card
    controler.render_segment = types.MethodType(
        Controler.render_segment, controler
    )
    controler._segment_tick_reference = types.MethodType(
        Controler._segment_tick_reference, controler
    )
    before = row.img.object

    row._commit_bounds(row.idx, (1.42, 1.91))

    assert row.img.object is not before
    assert "0.49" in row.duration_pane.object


def test_a_redraw_that_fails_leaves_the_card_alone(card):
    row, controler = card
    def boom(*a, **k):
        raise OSError("unreadable")
    controler.render_segment = boom
    before = row.img.object

    row._commit_bounds(row.idx, (1.42, 1.91))

    assert row.img.object is before
    assert row.bounds_correction == pytest.approx((1.42, 1.91))
