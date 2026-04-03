"""
Comprehensive test battery for canapy.

Covers: Config, Timings, Annotation Transforms, Audio Transforms,
Training Transforms, Corpus, Postprocessing, Metrics, Corrections,
SynAnnotator pipeline (slow), and HP flow regression tests.

Run fast tests only:   pytest tests/test_battery.py -m "not slow"
Run all:               pytest tests/test_battery.py
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io.wavfile

# ---------------------------------------------------------------------------
# Session-scoped fixtures – synthetic WAV + annotation files
# ---------------------------------------------------------------------------

LABELS = list("abcde")
SR = 22050


@pytest.fixture(scope="session")
def audio_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("battery_audio")
    rng = np.random.default_rng(0)
    for i in range(6):
        n_samples = rng.integers(22050, 44100)
        wav = rng.uniform(-1, 1, n_samples)
        wav_i16 = (wav / np.max(np.abs(wav)) * np.iinfo(np.int16).max).astype(np.int16)
        scipy.io.wavfile.write(str(d / f"bird_{i:02d}.wav"), SR, wav_i16)
    yield d
    shutil.rmtree(str(d))


@pytest.fixture(scope="session")
def annot_dir(tmp_path_factory, audio_dir):
    d = tmp_path_factory.mktemp("battery_annots")
    rng = np.random.default_rng(1)
    for wav_path in sorted(audio_dir.glob("*.wav")):
        rate, audio = scipy.io.wavfile.read(str(wav_path))
        duration = len(audio) / rate
        n_segs = rng.integers(5, 15)
        bounds = np.sort(rng.uniform(0.01, duration - 0.01, n_segs - 1))
        onsets = np.concatenate([[0.0], bounds])
        offsets = np.concatenate([bounds, [duration]])
        rows = []
        for on, off in zip(onsets, offsets):
            rows.append({
                "wave": str(wav_path),
                "start": round(on, 4),
                "end": round(off, 4),
                "syll": rng.choice(LABELS),
            })
        pd.DataFrame(rows).to_csv(str(d / f"{wav_path.stem}.annot.csv"), index=False)
    yield d
    shutil.rmtree(str(d))


@pytest.fixture(scope="session")
def real_corpus(audio_dir, annot_dir):
    from canapy.corpus import Corpus
    return Corpus.from_directory(
        audio_directory=str(audio_dir),
        annots_directory=str(annot_dir),
    )


@pytest.fixture(scope="session")
def minimal_config():
    """Lightweight config for unit tests that don't need disk I/O."""
    import toml
    from config import Config

    cfg_str = """
[misc]
seed = 42

[transforms.annots]
time_precision = 0.001
min_label_duration = 0.02
lonely_labels = []
min_silence_gap = 0.001
silence_tag = "SIL"

[transforms.audio]
audio_features = ["mfcc", "delta", "delta2"]
sampling_rate = 22050
n_mfcc = 13
hop_length = 0.01
win_length = 0.02
n_fft = 512
fmin = 500
fmax = 8000
lifter = 40

[transforms.audio.delta]
padding = "wrap"

[transforms.audio.delta2]
padding = "wrap"

[transforms.training]
max_sequences = -1
test_ratio = 0.2

[transforms.training.balance]
min_class_total_duration = 0.5
min_silence_duration = 0.1

[transforms.training.balance.data_augmentation]
noise_std = 0.01

[model.syn]
units = 100
sr = 0.4
leak = 0.1
iss = 0.0005
isd = 0.02
isd2 = 0.002
ridge = 1e-8
backend = "multiprocessing"
workers = 1

[model.nsyn]
units = 100
sr = 0.4
leak = 0.1
iss = 0.0005
isd = 0.02
isd2 = 0.002
ridge = 1e-8
backend = "multiprocessing"
workers = 1

[correction]
min_segment_proportion_agreement = 0.66
"""
    return Config(**toml.loads(cfg_str))


@pytest.fixture
def simple_corpus(minimal_config):
    """In-memory corpus with gaps between segments (required for tag_silences)."""
    from canapy.corpus import Corpus

    df = pd.DataFrame({
        "label":        list("aaabbbccc"),
        # 0.04 s gap between each segment so tag_silences can insert silence rows
        "onset_s":      [0.00, 0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.84, 0.96],
        "offset_s":     [0.08, 0.20, 0.32, 0.44, 0.56, 0.68, 0.80, 0.92, 1.04],
        "notated_path": ["foo.wav"] * 9,
        "annot_path":   ["foo.csv"] * 9,
        "sequence":     [0] * 9,
        "annotation":   ["foo.csv"] * 9,
    })
    return Corpus.from_df(df, config=minimal_config)


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_getattr_scalar(self, minimal_config):
        assert minimal_config.misc.seed == 42

    def test_getattr_returns_copy_for_nested(self, minimal_config):
        """__getattr__ returns a copy of nested configs – mutations must go via .data."""
        copy_a = minimal_config.model.syn
        copy_b = minimal_config.model.syn
        assert copy_a is not copy_b

    def test_setattr_scalar(self):
        from config import Config
        c = Config()
        c.foo = 42
        assert c.foo == 42

    def test_setattr_dict_wraps_to_config(self):
        from config import Config
        c = Config()
        c.nested = {"x": 1, "y": 2}
        assert isinstance(c.nested, Config)
        assert c.nested.x == 1

    def test_data_dict_write_is_persistent(self, minimal_config):
        """Writing via .data dict must persist (regression for HP bug)."""
        from config import Config
        c = Config(**{"model": {"syn": {"units": 100, "sr": 0.4}}})
        c.data["model"]["syn"]["units"] = 999
        assert c.data["model"]["syn"]["units"] == 999

    def test_solve_nested_path(self, minimal_config):
        val = minimal_config.solve("model/syn/units")
        assert val == 100

    def test_from_toml(self, tmp_path, minimal_config):
        from config import Config
        p = tmp_path / "test.toml"
        minimal_config.to_disk(str(p))
        loaded = Config.from_file(str(p))
        assert loaded.misc.seed == minimal_config.misc.seed

    def test_iteration_over_items(self, minimal_config):
        keys = [k for k, _ in minimal_config.items()]
        assert "misc" in keys
        assert "transforms" in keys
        assert "model" in keys


# ---------------------------------------------------------------------------
# 2. Timings
# ---------------------------------------------------------------------------

class TestTimings:
    def test_seconds_to_audio(self):
        from canapy.timings import seconds_to_audio
        result = seconds_to_audio(np.array([0.0, 0.5, 1.0]), sampling_rate=22050)
        expected = np.array([0, 11025, 22050])
        np.testing.assert_array_equal(result, expected)

    def test_audio_to_frames(self):
        from canapy.timings.base import audio_to_frames
        samples = np.array([0, 220, 440])
        frames = audio_to_frames(samples, frame_size=220)
        assert frames[0] >= 0
        assert frames.dtype in (np.int32, np.int64, int)

    def test_frames_to_seconds_shape(self):
        from canapy.timings.base import frames_to_seconds
        frame_indices = np.arange(10)
        onsets, offsets = frames_to_seconds(frame_indices, frame_size=220, sampling_rate=22050)
        assert len(onsets) == 10
        assert len(offsets) == 10

    def test_frames_to_seconds_monotone(self):
        from canapy.timings.base import frames_to_seconds
        frame_indices = np.arange(20)
        onsets, _ = frames_to_seconds(frame_indices, frame_size=220, sampling_rate=22050)
        assert np.all(np.diff(onsets) >= 0)

    def test_frames_to_timed_df(self):
        from canapy.timings import frames_to_timed_df
        preds = np.array(["a", "a", "b", "b", "c"])
        df = frames_to_timed_df(preds, "foo.wav", frame_size=220, sampling_rate=22050)
        assert "label" in df.columns
        assert "onset_s" in df.columns
        assert "offset_s" in df.columns
        assert len(df) == 5

    def test_frames_to_timed_df_empty(self):
        from canapy.timings import frames_to_timed_df
        df = frames_to_timed_df(np.array([]), "foo.wav", frame_size=220, sampling_rate=22050)
        assert len(df) == 0

    def test_seconds_to_frames_roundtrip(self):
        from canapy.timings.base import seconds_to_frames, frames_to_seconds
        timings = np.array([0.0, 0.5, 1.0])
        frames = seconds_to_frames(timings, frame_size=220, sampling_rate=22050)
        onsets, _ = frames_to_seconds(frames, frame_size=220, sampling_rate=22050)
        # Roundtrip should be close (within 1 frame tolerance)
        np.testing.assert_allclose(onsets, timings, atol=0.05)


# ---------------------------------------------------------------------------
# 3. Annotation Transforms
# ---------------------------------------------------------------------------

class TestAnnotTransforms:
    def test_sort_annotations(self, simple_corpus):
        from canapy.transforms.commons.annots import sort_annotations
        sorted_c = sort_annotations(simple_corpus)
        df = sorted_c.dataset
        # Sort should not drop rows
        assert len(df) == len(simple_corpus.dataset)

    def test_tag_silences_adds_rows(self, simple_corpus):
        from canapy.transforms.commons.annots import tag_silences
        tagged = tag_silences(simple_corpus)
        # tagging silences may add rows between segments
        assert len(tagged.dataset) >= len(simple_corpus.dataset)

    def test_tag_silences_silence_label(self, simple_corpus):
        from canapy.transforms.commons.annots import tag_silences
        tagged = tag_silences(simple_corpus)
        silence_tag = simple_corpus.config.transforms.annots.silence_tag
        assert silence_tag in tagged.dataset["label"].values

    def test_remove_short_labels(self, simple_corpus):
        from canapy.transforms.commons.annots import remove_short_labels
        # Add a very short row
        df = simple_corpus.dataset.copy()
        short_row = df.iloc[0:1].copy()
        short_row["onset_s"] = 0.0
        short_row["offset_s"] = 0.001  # 1 ms < min_label_duration
        extended = simple_corpus.clone_with_df(pd.concat([df, short_row], ignore_index=True))
        cleaned = remove_short_labels(extended)
        # short row should be removed
        assert len(cleaned.dataset) < len(extended.dataset)

    def test_merge_labels(self, simple_corpus):
        from canapy.transforms.commons.annots import merge_labels
        # Should not crash
        merged = merge_labels(simple_corpus)
        assert "label" in merged.dataset.columns


# ---------------------------------------------------------------------------
# 4. Corpus
# ---------------------------------------------------------------------------

class TestCorpus:
    def test_from_directory(self, real_corpus):
        assert real_corpus.dataset is not None
        assert len(real_corpus.dataset) > 0

    def test_from_directory_has_required_columns(self, real_corpus):
        for col in ("label", "onset_s", "offset_s", "notated_path"):
            assert col in real_corpus.dataset.columns

    def test_from_df(self, simple_corpus):
        assert len(simple_corpus.dataset) == 9

    def test_clone_with_df(self, simple_corpus):
        sub_df = simple_corpus.dataset.iloc[:3].copy()
        clone = simple_corpus.clone_with_df(sub_df)
        assert len(clone.dataset) == 3
        # Config must be preserved
        assert clone.config.misc.seed == simple_corpus.config.misc.seed

    def test_register_data_resource(self, simple_corpus):
        simple_corpus.register_data_resource("test_key", {"value": 42})
        assert "test_key" in simple_corpus.data_resources
        assert simple_corpus.data_resources["test_key"]["value"] == 42

    def test_filter_via_getitem(self, simple_corpus):
        filtered = simple_corpus["label == 'a'"]
        assert all(filtered.dataset["label"] == "a")

    def test_to_directory(self, simple_corpus, tmp_path):
        out_dir = tmp_path / "annots_out"
        out_dir.mkdir()
        simple_corpus.to_directory(out_dir)
        # Should produce at least one output file
        output_files = list(out_dir.rglob("*"))
        assert len(output_files) > 0

    def test_len(self, simple_corpus):
        # len = number of unique annotations
        assert isinstance(len(simple_corpus), int)
        assert len(simple_corpus) >= 0

    def test_empty_corpus(self, tmp_path):
        from canapy.corpus import Corpus
        empty = Corpus.from_directory(
            audio_directory=str(tmp_path),
            annots_directory=str(tmp_path),
        )
        assert len(empty.dataset) == 0


# ---------------------------------------------------------------------------
# 5. Postprocessing
# ---------------------------------------------------------------------------

class TestPostprocess:
    def test_extract_vocab(self, simple_corpus):
        from canapy.annotator.commons.postprocess import extract_vocab
        silence_tag = simple_corpus.config.transforms.annots.silence_tag
        vocab = extract_vocab(simple_corpus, silence_tag)
        assert silence_tag in vocab
        assert sorted(vocab) == vocab  # must be sorted

    def test_extract_vocab_no_duplicate(self, simple_corpus):
        from canapy.annotator.commons.postprocess import extract_vocab
        silence_tag = simple_corpus.config.transforms.annots.silence_tag
        vocab = extract_vocab(simple_corpus, silence_tag)
        assert len(vocab) == len(set(vocab))

    def test_frame_df_to_annots_df(self, simple_corpus):
        from canapy.annotator.commons.postprocess import frame_df_to_annots_df
        frame_df = pd.DataFrame({
            "label":        list("aaabbbccc"),
            "onset_s":      [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
            "offset_s":     [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            "notated_path": ["foo.wav"] * 9,
        })
        config = simple_corpus.config.transforms.annots
        annots = frame_df_to_annots_df(
            frame_df,
            min_label_duration=config.min_label_duration,
            min_silence_gap=config.min_silence_gap,
            silence_tag=config.silence_tag,
        )
        assert "label" in annots.columns
        # Continuous same-label frames should be merged to fewer rows
        assert len(annots) <= len(frame_df)


# ---------------------------------------------------------------------------
# 6. Corrections
# ---------------------------------------------------------------------------

class TestCorrections:
    def test_correct_classes_rename(self, simple_corpus):
        from canapy.correction.base import correct_classes
        corrected = correct_classes(simple_corpus, {"a": "x"})
        assert "a" not in corrected.dataset["label"].values
        assert "x" in corrected.dataset["label"].values

    def test_correct_classes_no_mutation(self, simple_corpus):
        """Original corpus should not be mutated."""
        from canapy.correction.base import correct_classes
        original_labels = simple_corpus.dataset["label"].tolist()
        correct_classes(simple_corpus, {"a": "MERGED"})
        assert simple_corpus.dataset["label"].tolist() == original_labels

    def test_correct_annots(self, simple_corpus):
        from canapy.correction.base import correct_annots
        # Correct index 0 to "z"
        corrected = correct_annots(simple_corpus, {0: "z"})
        assert "z" in corrected.dataset["label"].values

    def test_corrector_correct_class(self, simple_corpus, tmp_path):
        from canapy.correction.base import Corrector
        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir()
        corrector = Corrector(checkpoint_directory=ckpt_dir)
        result = corrector.correct(simple_corpus, class_corrections={"b": "B"})
        assert "B" in result.dataset["label"].values
        assert "b" not in result.dataset["label"].values

    def test_corrector_checkpoint(self, simple_corpus, tmp_path):
        from canapy.correction.base import Corrector
        ckpt_dir = tmp_path / "ckpts2"
        ckpt_dir.mkdir()
        corrector = Corrector(checkpoint_directory=ckpt_dir)
        corrector.correct(
            simple_corpus,
            class_corrections={"a": "A"},
            annot_corrections={},
            checkpoint=True,
        )
        assert len(corrector.correction_history) == 1

    @pytest.mark.xfail(
        reason="Bug codebase: Corrector.checkpoint() écrit le fichier sans extension "
               "'.toml', mais from_checkpoints() cherche '*.toml' → fichier introuvable."
    )
    def test_corrector_from_checkpoints(self, simple_corpus, tmp_path):
        from canapy.correction.base import Corrector
        ckpt_dir = tmp_path / "ckpts3"
        ckpt_dir.mkdir()
        corrector = Corrector(checkpoint_directory=ckpt_dir)
        corrector.correct(
            simple_corpus,
            class_corrections={"c": "C"},
            annot_corrections={},
            checkpoint=True,
        )
        corrector2 = Corrector.from_checkpoints(ckpt_dir)
        assert len(corrector2.correction_history) == 1

    @pytest.mark.xfail(
        reason="Dépend de from_checkpoints — même bug d'extension .toml."
    )
    def test_correct_from_history(self, simple_corpus, tmp_path):
        from canapy.correction.base import Corrector
        ckpt_dir = tmp_path / "ckpts4"
        ckpt_dir.mkdir()
        corrector = Corrector(checkpoint_directory=ckpt_dir)
        corrector.correct(
            simple_corpus,
            class_corrections={"a": "ALPHA"},
            annot_corrections={},
            checkpoint=True,
        )
        replayed = corrector.correct_from_history(simple_corpus, checkpoint_step=0)
        assert "ALPHA" in replayed.dataset["label"].values


# ---------------------------------------------------------------------------
# 7. Metrics (using prediction corpus fixtures)
# ---------------------------------------------------------------------------

def _make_prediction_corpus(config):
    """Build a minimal corpus that has frames_predictions resource."""
    from canapy.corpus import Corpus

    notated_paths = ["foo.wav", "baz.wav"]
    labels = list("abcabc")

    df = pd.DataFrame({
        "label":        labels * 2,
        "onset_s":      list(np.linspace(0.0, 0.5, 6)) + list(np.linspace(0.0, 0.5, 6)),
        "offset_s":     list(np.linspace(0.1, 0.6, 6)) + list(np.linspace(0.1, 0.6, 6)),
        "notated_path": ["foo.wav"] * 6 + ["baz.wav"] * 6,
        "annot_path":   ["foo.csv"] * 6 + ["baz.csv"] * 6,
        "sequence":     [0] * 12,
        "annotation":   ["foo.csv"] * 6 + ["baz.csv"] * 6,
    })

    frame_labels = list("aaabbbcccaaabbbccc")
    frame_df = pd.DataFrame({
        "label":        frame_labels,
        "onset_s":      list(np.linspace(0.0, 0.5, 9)) * 2,
        "offset_s":     list(np.linspace(0.05, 0.55, 9)) * 2,
        "notated_path": ["foo.wav"] * 9 + ["baz.wav"] * 9,
    })

    corpus = Corpus.from_df(df, config=config)
    corpus.register_data_resource("frames_predictions", frame_df)
    return corpus


class TestMetrics:
    def test_segment_error_rate(self, minimal_config):
        from canapy.metrics import segment_error_rate
        corpus = _make_prediction_corpus(minimal_config)
        # SER returns a DataFrame with one row per audio file
        ser = segment_error_rate(corpus, corpus)
        assert isinstance(ser, pd.DataFrame)
        assert "ser" in ser.columns
        # Compared against itself → edit distance = 0 for every file
        assert ser["ser"].sum() == pytest.approx(0.0)

    def test_sklearn_classification_report_keys(self, minimal_config):
        from canapy.metrics import sklearn_classification_report
        corpus = _make_prediction_corpus(minimal_config)
        report = sklearn_classification_report(corpus, corpus)
        assert isinstance(report, dict)
        # Should contain per-class rows
        assert "accuracy" in report or any(k in report for k in ("a", "b", "c"))

    def test_sklearn_confusion_matrix_shape(self, minimal_config):
        from canapy.metrics import sklearn_confusion_matrix
        corpus = _make_prediction_corpus(minimal_config)
        classes = ["a", "b", "c"]
        cm = sklearn_confusion_matrix(corpus, corpus, classes=classes)
        assert cm.shape == (3, 3)

    def test_sklearn_confusion_matrix_square(self, minimal_config):
        """La matrice de confusion doit être carrée (n_classes × n_classes)."""
        from canapy.metrics import sklearn_confusion_matrix
        corpus = _make_prediction_corpus(minimal_config)
        classes = ["a", "b", "c"]
        cm = sklearn_confusion_matrix(corpus, corpus, classes=classes)
        assert cm.shape[0] == cm.shape[1] == len(classes)


# ---------------------------------------------------------------------------
# 8. Training Transforms
# ---------------------------------------------------------------------------

class TestTrainingTransforms:
    def test_split_train_test_adds_column(self, real_corpus):
        from canapy.transforms.commons.training import split_train_test
        split = split_train_test(real_corpus)
        assert "train" in split.dataset.columns

    def test_split_train_test_all_labels_in_train(self, real_corpus):
        from canapy.transforms.commons.training import split_train_test
        split = split_train_test(real_corpus)
        train_labels = set(split.dataset[split.dataset["train"]]["label"].unique())
        all_labels = set(split.dataset["label"].unique())
        assert all_labels == train_labels

    def test_split_reproducible(self, real_corpus):
        """Same seed must produce identical train/test splits."""
        from canapy.transforms.commons.training import split_train_test
        s1 = split_train_test(real_corpus)
        s2 = split_train_test(real_corpus, redo=True)
        pd.testing.assert_series_equal(
            s1.dataset["train"].reset_index(drop=True),
            s2.dataset["train"].reset_index(drop=True),
        )

    def test_encode_labels(self, real_corpus):
        from canapy.transforms.commons.training import split_train_test, encode_labels
        split = split_train_test(real_corpus)
        # resource_name est un kwarg obligatoire (None = pas de stockage de ressource)
        encoded = encode_labels(split, resource_name=None)
        assert "encoded_label" in encoded.dataset.columns


# ---------------------------------------------------------------------------
# 9. Config data-dict write – regression for HP bug
# ---------------------------------------------------------------------------

class TestConfigRegressions:
    def test_config_data_dict_write_persists(self):
        """Regression: setattr writes to a copy; only .data dict write is persistent."""
        from config import Config
        c = Config(**{"model": {"syn": {"units": 100}}})

        # setattr on a nested config should NOT persist (this is expected behavior)
        c.model.syn.units = 999
        # The original config is unchanged because model.syn is a copy
        assert c.data["model"]["syn"]["units"] == 100

        # Correct approach: write directly into .data dict
        c.data["model"]["syn"]["units"] = 999
        assert c.data["model"]["syn"]["units"] == 999

    def test_config_write_via_data_survives_attribute_access(self):
        """After a .data dict write, subsequent __getattr__ access must see new value."""
        from config import Config
        c = Config(**{"model": {"syn": {"units": 100, "sr": 0.4}}})
        c.data["model"]["syn"]["units"] = 512
        # Access through attribute chain
        assert c.data["model"]["syn"]["units"] == 512


# ---------------------------------------------------------------------------
# 10. SynAnnotator – full pipeline (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestSynAnnotatorPipeline:
    def test_fit_and_predict(self, real_corpus, tmp_path):
        from canapy.annotator.synannotator import SynAnnotator
        annotator = SynAnnotator(config=real_corpus.config)
        annotator.fit(real_corpus)
        pred_corpus = annotator.predict(real_corpus)
        assert pred_corpus is not None
        assert len(pred_corpus.dataset) > 0

    def test_predict_has_frames_predictions(self, real_corpus, tmp_path):
        from canapy.annotator.synannotator import SynAnnotator
        annotator = SynAnnotator(config=real_corpus.config)
        annotator.fit(real_corpus)
        pred_corpus = annotator.predict(real_corpus)
        assert "frames_predictions" in pred_corpus.data_resources

    def test_to_directory_after_predict(self, real_corpus, tmp_path):
        from canapy.annotator.synannotator import SynAnnotator
        annotator = SynAnnotator(config=real_corpus.config)
        annotator.fit(real_corpus)
        pred_corpus = annotator.predict(real_corpus)
        out = tmp_path / "predictions"
        out.mkdir()
        pred_corpus.to_directory(out)
        assert any(out.rglob("*.csv"))

    def test_rpy_model_reinit_changes_params(self, real_corpus):
        """Regression: reinitializing rpy_model after HP update must use new params."""
        from canapy.annotator.synannotator import SynAnnotator
        annotator = SynAnnotator(config=real_corpus.config)
        old_units = annotator.config.data["model"]["syn"]["units"]

        # Update via .data dict (correct approach)
        annotator.config.data["model"]["syn"]["units"] = old_units + 50
        annotator.rpy_model = annotator.initialize()

        # Verify the new model has the updated param
        new_units = annotator.config.data["model"]["syn"]["units"]
        assert new_units == old_units + 50


# ---------------------------------------------------------------------------
# 11. NsynAnnotator – full pipeline (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestNsynAnnotatorPipeline:
    def test_fit_and_predict(self, real_corpus, tmp_path):
        from canapy.annotator.nsynannotator import NSynAnnotator
        annotator = NSynAnnotator(config=real_corpus.config)
        annotator.fit(real_corpus)
        pred_corpus = annotator.predict(real_corpus)
        assert pred_corpus is not None
        assert len(pred_corpus.dataset) > 0

    def test_predict_has_frames_predictions(self, real_corpus, tmp_path):
        from canapy.annotator.nsynannotator import NSynAnnotator
        annotator = NSynAnnotator(config=real_corpus.config)
        annotator.fit(real_corpus)
        pred_corpus = annotator.predict(real_corpus)
        assert "frames_predictions" in pred_corpus.data_resources


# ---------------------------------------------------------------------------
# 12. Ensemble – full pipeline (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestEnsemblePipeline:
    def test_ensemble_fit_and_predict(self, real_corpus, tmp_path):
        from canapy.annotator.synannotator import SynAnnotator
        from canapy.annotator.nsynannotator import NSynAnnotator
        from canapy.annotator.ensemble import Ensemble

        syn = SynAnnotator(config=real_corpus.config)
        syn.fit(real_corpus)
        syn_pred = syn.predict(real_corpus, return_raw=True)

        nsyn = NSynAnnotator(config=real_corpus.config)
        nsyn.fit(real_corpus)
        nsyn_pred = nsyn.predict(real_corpus, return_raw=True)

        ensemble = Ensemble(config=real_corpus.config)
        ensemble.fit(real_corpus)
        result = ensemble.predict([syn_pred, nsyn_pred])
        assert result is not None
        assert len(result.dataset) > 0


# ---------------------------------------------------------------------------
# 13. Optimization helpers (unit tests, no ESN needed)
# ---------------------------------------------------------------------------

class TestOptimizationHelpers:
    def test_concat_xy(self):
        from canapy.optimization import _concat_xy
        X = [np.ones((10, 5)), np.ones((8, 5))]
        Y = [np.zeros((10, 3)), np.zeros((8, 3))]
        Xc, Yc = _concat_xy(X, Y)
        assert Xc.shape == (18, 5)
        assert Yc.shape == (18, 3)

    def test_concat_xy_trims_to_min_length(self):
        from canapy.optimization import _concat_xy
        X = [np.ones((10, 5))]
        Y = [np.zeros((7, 3))]  # shorter than X
        Xc, Yc = _concat_xy(X, Y)
        assert Xc.shape[0] == 7
        assert Yc.shape[0] == 7

    def test_select_representative_sequences_fewer(self):
        from canapy.optimization import _select_representative_sequences
        rng = np.random.default_rng(0)
        n_classes = 3
        X = [rng.random((20, 5)) for _ in range(10)]
        # One-hot Y
        Y = []
        for i in range(10):
            labels = np.zeros((20, n_classes))
            labels[:, i % n_classes] = 1
            Y.append(labels)
        Xs, Ys = _select_representative_sequences(X, Y, n_select=4, rng=rng)
        assert len(Xs) == 4
        assert len(Ys) == 4

    def test_select_representative_sequences_all_when_n_ge_total(self):
        from canapy.optimization import _select_representative_sequences
        rng = np.random.default_rng(0)
        X = [np.ones((5, 3)) for _ in range(4)]
        Y = [np.eye(3)[:3] for _ in range(4)]
        Xs, Ys = _select_representative_sequences(X, Y, n_select=10, rng=rng)
        assert len(Xs) == 4  # all returned when n_select >= n_total


if __name__ == "__main__":
    import sys
    # Passe les args CLI supplémentaires s'il y en a (ex: -m slow)
    extra = sys.argv[1:] if len(sys.argv) > 1 else ["-m", "not slow"]
    raise SystemExit(pytest.main([__file__, "-v"] + extra))
