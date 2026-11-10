# Licence: BSD-3-Clause
"""A correction must invalidate what training derived from the old annotations."""
import types

import pandas as pd
import pytest

import dashboard.controler.base as base
from canapy.corpus import Corpus
from canapy.correction import make_annot_key
from dashboard.controler.base import Controler, TRAINING_DERIVED_RESOURCES

PATH = "/song.wav"


@pytest.fixture()
def controler(config):
    df = pd.DataFrame({
        "label": ["a", "b", "c"],
        "onset_s": [0.0, 1.0, 2.0],
        "offset_s": [0.5, 1.5, 2.5],
        "notated_path": [PATH] * 3,
        # real corpora carry these; merge_labels and the training pipeline
        # sort on them
        "annotation": [0] * 3,
        "sequence": [0] * 3,
    })
    corpus = Corpus.from_df(df, config=config)
    # what a first training leaves behind
    for name in TRAINING_DERIVED_RESOURCES:
        corpus.register_data_resource(name, df.copy())
    corpus.register_data_resource("syn_mfcc", df.copy())

    ctl = types.SimpleNamespace(
        config=config, corpus=corpus, compute_classes=lambda: None,
        _correction_store={"class": {}, "annot": {}, "bounds": {}},
        _metrics_store={"misclass": None},
    )
    for name in ("apply_live_corrections", "upload_corrections",
                 "_drop_stale_training_resources"):
        setattr(ctl, name, types.MethodType(getattr(Controler, name), ctl))
    return ctl


def _key(onset):
    return make_annot_key(PATH, onset)


@pytest.mark.parametrize("target, corrections", [
    ("bounds", {"__key__": (0.9, 1.7)}),
    ("annot", {"__key__": "z"}),
    ("class", {"b": "a"}),
])
def test_every_channel_forgets_the_stale_training_data(controler, target, corrections):
    corrections = {(_key(1.0) if k == "__key__" else k): v
                   for k, v in corrections.items()}

    controler.apply_live_corrections(corrections, target)

    assert not any(name in controler.corpus.data_resources
                   for name in TRAINING_DERIVED_RESOURCES)


def test_the_raw_mfccs_are_kept(controler):
    # per file, and blind to the annotations: a correction leaves them valid
    controler.apply_live_corrections({_key(1.0): (0.9, 1.7)}, "bounds")
    assert "syn_mfcc" in controler.corpus.data_resources


def test_training_sees_the_correction_instead_of_the_stale_copy(controler):
    from canapy.transforms.commons.training import prepare_dataset_for_training

    stale = controler.corpus.data_resources["dataset"]
    controler.apply_live_corrections({_key(1.0): (0.9, 1.7)}, "bounds")
    prepared = prepare_dataset_for_training(controler.corpus)

    # without the invalidation this returns the corpus untouched, still
    # carrying the bounds the annotation had before the correction
    assert prepared is not controler.corpus
    moved = prepared.dataset[prepared.dataset.label == "b"].iloc[0]
    assert (moved.onset_s, moved.offset_s) == pytest.approx((0.9, 1.7))
    assert stale[stale.label == "b"].iloc[0].onset_s == pytest.approx(1.0)


def test_without_the_invalidation_training_would_see_the_old_bounds(controler):
    from canapy.transforms.commons.training import prepare_dataset_for_training

    controler._drop_stale_training_resources = lambda: None
    controler.apply_live_corrections({_key(1.0): (0.9, 1.7)}, "bounds")

    assert prepare_dataset_for_training(controler.corpus) is controler.corpus


def test_a_corpus_without_those_resources_is_left_alone(controler):
    for name in TRAINING_DERIVED_RESOURCES:
        del controler.corpus.data_resources[name]
    controler.apply_live_corrections({_key(1.0): (0.9, 1.7)}, "bounds")
    assert "syn_mfcc" in controler.corpus.data_resources
