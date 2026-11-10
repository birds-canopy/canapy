# Licence: BSD-3-Clause
"""From a validated boundary to the corrected corpus, on the eval page."""
import types

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from canapy.corpus import Corpus
from canapy.correction import make_annot_key
from dashboard.controler.base import Controler
from dashboard.view.eval.samplecorrection import (
    SampleCorrectionDashboard, SampleCorrectorView,
)

SR = 44100


@pytest.fixture()
def song(tmp_path):
    rng = np.random.default_rng(0)
    sf.write(str(tmp_path / "song.wav"), rng.normal(0, 1e-3, int(3.0 * SR)), SR)
    return str(tmp_path / "song.wav")


@pytest.fixture()
def dashboard(song, config):
    df = pd.DataFrame({
        "label": ["A", "B"],
        "onset_s": [1.5, 2.0], "offset_s": [1.8, 2.4],
        "notated_path": [song] * 2,
        "annotation": [0, 0], "sequence": [0, 0],
    })
    corpus = Corpus.from_df(df, config=config)
    misclass = df.copy()
    misclass["pred_syn-esn"] = "B"

    ctl = types.SimpleNamespace(
        config=config, corpus=corpus, compute_classes=lambda: None,
        _correction_store={"class": {}, "annot": {}, "bounds": {}},
        _metrics_store={"misclass": misclass},
    )
    ctl.misclassified_segments = misclass
    for name in ("apply_live_corrections", "upload_corrections",
                 "_drop_stale_training_resources", "load_repertoire",
                 "_segment_tick_reference", "render_segment",
                 "_evict_repertoire_cache"):
        setattr(ctl, name, types.MethodType(getattr(Controler, name), ctl))
    ctl._repertoire_cache = {}

    dash = types.SimpleNamespace(
        controler=ctl, registry={"sample": {}, "class": {}}, total_corrected=0,
        listen_correction=lambda label, increment: None,
        check_correction=lambda value: True,
        open_modal=lambda content: None, close_modal=lambda: None,
        save_msg=types.SimpleNamespace(object="", visible=False),
        _refresh_after_annot_save=lambda: None,
    )
    view = SampleCorrectorView(dash, "A")
    dash.registry["sample"]["A"] = view
    return dash, view, ctl


def test_a_validated_boundary_is_collected_by_the_view(dashboard, song):
    dash, view, _ = dashboard
    card = list(view.registry.values())[0]

    card._commit_bounds(card.idx, (1.42, 1.91))

    assert view.bounds_corrections == {make_annot_key(song, 1.5): (1.42, 1.91)}


def test_saving_moves_the_annotation_in_the_corpus(dashboard):
    dash, view, ctl = dashboard
    card = list(view.registry.values())[0]
    card._commit_bounds(card.idx, (1.42, 1.91))

    SampleCorrectionDashboard.on_click_save(dash, None)

    moved = ctl.corpus.dataset[ctl.corpus.dataset.label == "A"].iloc[0]
    assert (moved.onset_s, moved.offset_s) == pytest.approx((1.42, 1.91))


def test_saving_with_nothing_pending_does_nothing(dashboard):
    dash, view, ctl = dashboard
    before = ctl.corpus.dataset.copy()

    SampleCorrectionDashboard.on_click_save(dash, None)

    assert list(ctl.corpus.dataset.onset_s) == list(before.onset_s)


def test_validating_alone_leaves_the_corpus_untouched(dashboard):
    # Validate only marks the correction pending — Save all is what writes it
    dash, view, ctl = dashboard
    card = list(view.registry.values())[0]

    card._commit_bounds(card.idx, (1.42, 1.91))

    unchanged = ctl.corpus.dataset[ctl.corpus.dataset.label == "A"].iloc[0]
    assert (unchanged.onset_s, unchanged.offset_s) == pytest.approx((1.5, 1.8))
