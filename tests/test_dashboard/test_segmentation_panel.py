# Licence: BSD-3-Clause
import types

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from canapy.corpus import Corpus
from canapy.segmentation import pools_period
from dashboard.view.segmentation import SyllableSegmentationPanel

SR = 44100
TONE_HZ = 4000.0
BURST_S = 0.040
GAP_S = 0.020
SILENCE_S = 0.200
PHRASES = [("A", 6), ("B", 4)]


@pytest.fixture()
def predictions(tmp_path, config):
    """One corpus of phrase-level predictions, as a model would return."""
    rng = np.random.default_rng(0)
    chunks, rows, cursor = [], [], 0.0

    def silence(duration):
        nonlocal cursor
        chunks.append(rng.normal(0, 1e-3, int(duration * SR)))
        cursor += duration

    silence(SILENCE_S)
    for label, n_bursts in PHRASES:
        onset = cursor
        for i in range(n_bursts):
            if i:
                silence(GAP_S)
            n = int(BURST_S * SR)
            t = np.arange(n) / SR
            chunks.append(np.sin(2 * np.pi * TONE_HZ * t) + rng.normal(0, 1e-3, n))
            cursor += BURST_S
        rows.append({"label": label, "onset_s": onset, "offset_s": cursor})
        silence(SILENCE_S)

    path = tmp_path / "song.wav"
    sf.write(str(path), np.concatenate(chunks), SR)

    df = pd.DataFrame(rows)
    df["notated_path"] = str(path)
    df["annot_path"] = str(tmp_path / "song.csv")
    return {"syn": Corpus.from_df(df, config=config)}


@pytest.fixture()
def owner(config):
    return types.SimpleNamespace(controler=types.SimpleNamespace(config=config))


def _ready(owner, predictions):
    panel = SyllableSegmentationPanel(owner)
    panel.set_predictions(predictions)
    panel._on_toggle(None)
    return panel


def test_panel_waits_for_predictions(owner):
    # segmentation runs on predictions, so there is nothing to show before
    panel = SyllableSegmentationPanel(owner)
    panel._on_toggle(None)

    assert panel._state == "waiting"
    assert not panel.controls.visible
    assert panel._figure is None
    assert not panel.enabled
    assert panel.layout.get_root() is not None


def test_predictions_open_the_panel(owner, predictions):
    panel = _ready(owner, predictions)

    assert panel._state == "ready"
    assert panel.controls.visible
    assert panel.model_select.options == ["syn"]
    assert panel._figure is not None
    assert panel.layout.get_root() is not None


def test_menu_lists_the_labels(owner, predictions):
    panel = SyllableSegmentationPanel(owner)
    assert panel.label_select.options == []

    panel.set_predictions(predictions)
    panel._on_toggle(None)
    assert panel.label_select.options == sorted(label for label, _ in PHRASES)


def test_opening_selection_is_reproducible(owner, predictions):
    first = _ready(owner, predictions)
    second = _ready(owner, predictions)
    assert first.label_select.value == second.label_select.value
    assert first._example["duration"] == second._example["duration"]


def test_selecting_another_label_redraws(owner, predictions):
    panel = _ready(owner, predictions)
    figure = panel._figure

    other = next(l for l in panel.label_select.options if l != panel._example["label"])
    panel.label_select.value = other

    assert panel._example["label"] == other
    assert len(panel._example["segments"]) == dict(PHRASES)[other]
    assert panel._figure is not figure
    assert len(panel.preview) == 1


def test_duration_setting_acts_on_the_same_phrase(owner, predictions):
    panel = _ready(owner, predictions)
    selected = panel.label_select.value
    phrase = panel._example
    assert len(phrase["segments"]) == dict(PHRASES)[phrase["label"]]

    panel.duration_input.value = 0.200
    assert panel.label_select.value == selected
    assert panel._example["duration"] == phrase["duration"]
    assert panel._example["segments"] == []


def test_settings_reach_the_config(owner, predictions):
    # a setting that only moved the preview would let the picture and the
    # exported file disagree
    panel = _ready(owner, predictions)
    section = owner.controler.config.data["segmentation"]

    panel.target_select.value = "repeated unit"
    assert section["group_syllables"] is True
    assert panel.pool_toggle.visible

    panel.pool_toggle.value = False
    assert section["pool_period"] is False

    panel.target_select.value = "acoustic element"
    assert section["group_syllables"] is False
    assert not panel.pool_toggle.visible


def test_opening_the_panel_leaves_the_config_alone(owner):
    owner.controler.config.data["segmentation"] = {
        "group_syllables": True, "pool_period": True,
    }

    panel = SyllableSegmentationPanel(owner)

    assert owner.controler.config.data["segmentation"]["pool_period"] is True
    assert panel.pool_toggle.value is True


def test_the_period_is_pooled_by_default(owner):
    assert pools_period(owner.controler.config) is True
    assert SyllableSegmentationPanel(owner).pool_toggle.value is True


def test_segmenting_needs_predictions(owner, predictions):
    panel = SyllableSegmentationPanel(owner)
    assert not panel.enabled          # no predictions yet

    panel.set_predictions(predictions)
    assert panel.enabled

    panel.reset()
    assert panel._state == "waiting"
    assert not panel.enabled


def test_duplicate_index_labels_still_draw(tmp_path, owner, config):
    # predictions are assembled one file at a time, each numbered from zero,
    # so the dataset arrives with a repeated index
    rng = np.random.default_rng(0)
    chunks, rows, cursor = [], [], 0.0

    def silence(d):
        nonlocal cursor
        chunks.append(rng.normal(0, 1e-3, int(d * SR)))
        cursor += d

    silence(SILENCE_S)
    for k in range(8):
        onset = cursor
        for i in range(2 + k % 4):        # phrases of different lengths
            if i:
                silence(GAP_S)
            n = int(BURST_S * SR)
            t = np.arange(n) / SR
            chunks.append(np.sin(2 * np.pi * TONE_HZ * t) + rng.normal(0, 1e-3, n))
            cursor += BURST_S
        rows.append({"label": "A", "onset_s": onset, "offset_s": cursor})
        silence(SILENCE_S)

    path = tmp_path / "song.wav"
    sf.write(str(path), np.concatenate(chunks), SR)
    df = pd.DataFrame(rows)
    df["notated_path"] = str(path)
    df["annot_path"] = str(tmp_path / "song.csv")
    corpus = Corpus.from_df(df, config=config)
    corpus.dataset.index = [i % 4 for i in range(len(corpus.dataset))]

    panel = SyllableSegmentationPanel(owner)
    panel.set_predictions({"syn": corpus})
    panel._on_toggle(None)

    assert panel._figure is not None
    assert panel.status.object == ""
    assert len(panel.preview) == 1


def test_already_segmented_predictions_are_refused(owner, predictions):
    panel = _ready(owner, predictions)
    panel.mark_unavailable()

    assert panel._state == "unavailable"
    assert not panel.controls.visible
    assert not panel.enabled


def test_the_controler_segments_a_session_corpus(predictions, config):
    """`Controler.segment_syllables`, which no page calls yet."""
    from dashboard.controler.base import Controler

    config.data.pop("segmentation")
    corpus = predictions["syn"]
    controler = types.SimpleNamespace(
        config=config, corpus=corpus, _repertoire_cache={},
        preprocess_done=True, fit_done=True, eval_done=True, export_done=True,
        compute_classes=lambda: None,
    )

    before, after = Controler.segment_syllables(controler)

    assert (before, after) == (len(corpus.dataset), sum(n for _, n in PHRASES))
    assert config.transforms.annots.min_label_duration < 0.02
    assert not controler.fit_done
