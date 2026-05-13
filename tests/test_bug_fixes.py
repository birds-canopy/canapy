"""
Targeted regression tests for bugs 1–7 fixed in annotator/base, transforms/annots,
postprocess, corpus and correction modules.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from config import default_config

from canapy.corpus import Corpus
from canapy.correction.base import Corrector
from canapy.transforms.commons.annots import merge_labels, remove_short_labels
from canapy.annotator.commons.postprocess import frame_df_to_annots_df
from canapy.annotator.synannotator import SynAnnotator
from canapy.annotator.nsynannotator import NSynAnnotator
from canapy.annotator.ensemble import Ensemble


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_corpus(rows, config=None):
    df = pd.DataFrame(rows)
    return Corpus.from_df(df, config=config or default_config)


# ---------------------------------------------------------------------------
# Bug 1 — merge_labels: gap direction is correct
# ---------------------------------------------------------------------------

class TestMergeLabelsBug1:
    """merge_labels must only fuse consecutive identical labels when the
    silence gap between them is <= min_silence_gap, not always."""

    def _corpus_with_gap(self, gap_s):
        # Two "A" syllables with a given gap between them
        rows = [
            {"label": "A", "onset_s": 0.0,           "offset_s": 0.1,           "notated_path": "f.wav", "annot_path": "f.csv", "sequence": 0, "annotation": "a"},
            {"label": "A", "onset_s": 0.1 + gap_s,   "offset_s": 0.1 + gap_s + 0.1, "notated_path": "f.wav", "annot_path": "f.csv", "sequence": 0, "annotation": "a"},
        ]
        return _make_corpus(rows)

    def test_small_gap_merges(self):
        """Gap <= min_silence_gap (0.001 s) → the two As must become one."""
        corpus = self._corpus_with_gap(0.0005)
        result = merge_labels(corpus)
        assert len(result.dataset) == 1, "Expected 1 merged annotation"
        assert result.dataset.iloc[0]["onset_s"] == pytest.approx(0.0)
        assert result.dataset.iloc[0]["offset_s"] == pytest.approx(0.2005)

    def test_large_gap_does_not_merge(self):
        """Gap >> min_silence_gap (5 s) → the two As must stay separate."""
        corpus = self._corpus_with_gap(5.0)
        result = merge_labels(corpus)
        assert len(result.dataset) == 2, (
            f"Expected 2 separate annotations but got {len(result.dataset)} — "
            "gap was incorrectly ignored (bug 1 not fixed)"
        )

    def test_different_labels_not_merged(self):
        """Different labels must never be merged."""
        rows = [
            {"label": "A", "onset_s": 0.0, "offset_s": 0.1, "notated_path": "f.wav", "annot_path": "f.csv", "sequence": 0, "annotation": "a"},
            {"label": "B", "onset_s": 0.1, "offset_s": 0.2, "notated_path": "f.wav", "annot_path": "f.csv", "sequence": 0, "annotation": "a"},
        ]
        corpus = _make_corpus(rows)
        result = merge_labels(corpus)
        assert len(result.dataset) == 2


# ---------------------------------------------------------------------------
# Bug 1 (bis) — frame_df_to_annots_df: same gap fix in postprocess
# ---------------------------------------------------------------------------

class TestPostprocessGapBug1:
    def test_large_gap_not_merged(self):
        frame_df = pd.DataFrame({
            "label":        ["A"] * 3 + ["SIL"] * 50 + ["A"] * 3,
            "onset_s":      [i * 0.01 for i in range(56)],
            "offset_s":     [i * 0.01 + 0.01 for i in range(56)],
            "notated_path": ["f.wav"] * 56,
        })
        result = frame_df_to_annots_df(
            frame_df,
            min_label_duration=0.02,
            min_silence_gap=0.001,
            silence_tag="SIL",
        )
        # After removing SIL, we should have 2 "A" segments, not one merged blob
        a_segs = result[result["label"] == "A"]
        assert len(a_segs) == 2, (
            f"Expected 2 separate A segments, got {len(a_segs)} — "
            "gap direction bug in postprocess not fixed"
        )


# ---------------------------------------------------------------------------
# Bug 2 — _vocab is per-instance, not shared across instances
# ---------------------------------------------------------------------------

class TestVocabIsolationBug2:
    def test_syn_vocab_isolated(self):
        a = SynAnnotator(config=default_config)
        b = SynAnnotator(config=default_config)
        a._vocab = ["X", "Y"]
        assert b._vocab == [], (
            "SynAnnotator instances share _vocab — class-level mutable default not fixed"
        )

    def test_nsyn_vocab_isolated(self):
        a = NSynAnnotator(config=default_config)
        b = NSynAnnotator(config=default_config)
        a._vocab.append("Z")
        assert b._vocab == [], (
            "NSynAnnotator instances share _vocab — class-level mutable default not fixed"
        )

    def test_ensemble_vocab_isolated(self):
        a = Ensemble(config=default_config)
        b = Ensemble(config=default_config)
        a._vocab = ["A"]
        assert b._vocab == []

    def test_trained_isolated(self):
        a = SynAnnotator(config=default_config)
        b = SynAnnotator(config=default_config)
        a._trained = True
        assert b._trained is False, (
            "SynAnnotator instances share _trained — class-level bool not fixed"
        )

    def test_cross_class_vocab_isolated(self):
        syn = SynAnnotator(config=default_config)
        nsyn = NSynAnnotator(config=default_config)
        syn._vocab = ["A", "B"]
        assert nsyn._vocab == [], "SynAnnotator and NSynAnnotator share _vocab"


# ---------------------------------------------------------------------------
# Bug 3 — correction_history is per-instance
# ---------------------------------------------------------------------------

class TestCorrectionHistoryIsolationBug3:
    def test_history_not_shared(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            c1 = Corrector(checkpoint_directory=d1)
            c2 = Corrector(checkpoint_directory=d2)
            c1.correction_history.append({"class": {}, "annot": {}})
            assert len(c2.correction_history) == 0, (
                "Corrector instances share correction_history — factory default not fixed"
            )


# ---------------------------------------------------------------------------
# Bug 4 — remove_short_labels: durations.round() result is assigned
# ---------------------------------------------------------------------------

class TestRemoveShortLabelsBug4:
    def test_borderline_duration_rounded_correctly(self):
        """A label whose raw float duration is 0.019999... should be treated as
        0.020 after rounding to time_precision=0.001 and thus NOT removed."""
        epsilon = 1e-10
        rows = [
            # onset/offset chosen so that offset - onset = 0.020 - epsilon (barely below 20ms)
            {"label": "A", "onset_s": 0.0, "offset_s": 0.02 - epsilon,
             "notated_path": "f.wav", "annot_path": "f.csv", "sequence": 0, "annotation": "a"},
        ]
        corpus = _make_corpus(rows)
        result = remove_short_labels(corpus)
        # After rounding to 1ms, 0.019999999... → 0.020 which equals min_label_duration
        # The comparison is strict <, so 0.020 should be kept.
        assert len(result.dataset) == 1, (
            "Label at exactly the threshold was incorrectly removed — "
            "round() result may not be assigned (bug 4)"
        )

    def test_clearly_short_label_removed(self):
        rows = [
            {"label": "A", "onset_s": 0.0, "offset_s": 0.005,
             "notated_path": "f.wav", "annot_path": "f.csv", "sequence": 0, "annotation": "a"},
        ]
        corpus = _make_corpus(rows)
        result = remove_short_labels(corpus)
        assert len(result.dataset) == 0


# ---------------------------------------------------------------------------
# Bug 5 — correct_from_history: off-by-one raises ValueError, not IndexError
# ---------------------------------------------------------------------------

class TestCorrectFromHistoryBug5:
    def test_out_of_range_raises_value_error(self):
        with tempfile.TemporaryDirectory() as d:
            corrector = Corrector(checkpoint_directory=d, correction_history=[
                {"class": {}, "annot": {}},
                {"class": {}, "annot": {}},
            ])
            rows = [{"label": "A", "onset_s": 0.0, "offset_s": 0.1,
                     "notated_path": "f.wav", "annot_path": "f.csv",
                     "sequence": 0, "annotation": "a"}]
            corpus = _make_corpus(rows)

            # len == 2, so valid indices are 0 and 1. Index 2 must raise ValueError.
            with pytest.raises(ValueError):
                corrector.correct_from_history(corpus, 2)

    def test_valid_last_index_works(self):
        with tempfile.TemporaryDirectory() as d:
            corrector = Corrector(checkpoint_directory=d, correction_history=[
                {"class": {"A": "B"}, "annot": {}},
            ])
            rows = [{"label": "A", "onset_s": 0.0, "offset_s": 0.1,
                     "notated_path": "f.wav", "annot_path": "f.csv",
                     "sequence": 0, "annotation": "a"}]
            corpus = _make_corpus(rows)
            result = corrector.correct_from_history(corpus, 0)
            assert result.dataset.iloc[0]["label"] == "B"


# ---------------------------------------------------------------------------
# Bug 6 — checkpoint writes .toml files that from_checkpoints can find
# ---------------------------------------------------------------------------

class TestCheckpointExtensionBug6:
    def test_checkpoint_creates_toml_file(self):
        with tempfile.TemporaryDirectory() as d:
            corrector = Corrector(checkpoint_directory=d)
            corrector.checkpoint({"class": {"A": "B"}, "annot": {}})
            toml_files = list(Path(d).glob("*.toml"))
            assert len(toml_files) == 1, (
                f"Expected 1 .toml checkpoint file, found {toml_files} — "
                ".toml extension missing from checkpoint filename (bug 6)"
            )

    def test_from_checkpoints_finds_saved_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            corrector = Corrector(checkpoint_directory=d)
            corrector.checkpoint({"class": {"A": "B"}, "annot": {}})
            corrector.checkpoint({"class": {"C": "D"}, "annot": {}})

            loaded = Corrector.from_checkpoints(d)
            assert len(loaded.correction_history) == 2, (
                f"Expected 2 checkpoints, found {len(loaded.correction_history)} — "
                "from_checkpoints can't find files (bug 6 not fully fixed)"
            )
            assert loaded.correction_history[0]["class"] == {"A": "B"}
            assert loaded.correction_history[1]["class"] == {"C": "D"}


# ---------------------------------------------------------------------------
# Bug 7 — checkpoint does not crash when annot_corrections is None
# ---------------------------------------------------------------------------

class TestCheckpointNoneBug7:
    def test_checkpoint_with_none_annots_does_not_crash(self):
        with tempfile.TemporaryDirectory() as d:
            corrector = Corrector(checkpoint_directory=d)
            rows = [{"label": "A", "onset_s": 0.0, "offset_s": 0.1,
                     "notated_path": "f.wav", "annot_path": "f.csv",
                     "sequence": 0, "annotation": "a"}]
            corpus = _make_corpus(rows)
            # Must not raise AttributeError: 'NoneType' object has no attribute 'items'
            result = corrector.correct(corpus, class_corrections={"A": "B"},
                                       annot_corrections=None, checkpoint=True)
            assert result.dataset.iloc[0]["label"] == "B"

    def test_checkpoint_file_content_with_none_annots(self):
        with tempfile.TemporaryDirectory() as d:
            corrector = Corrector(checkpoint_directory=d)
            corrector.checkpoint({"class": {"A": "B"}, "annot": None})
            loaded = Corrector.from_checkpoints(d)
            assert loaded.correction_history[0]["annot"] == {}
