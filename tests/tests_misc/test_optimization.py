# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
"""Tests for canapy/optimization.py."""

import tempfile

import numpy as np
import pandas as pd
import pytest
import toml

from config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df_prepared(seqids, labels_per_seq):
    return pd.DataFrame({
        "seqid": seqids,
        "label": labels_per_seq,
    })


def _make_npz(path, n_features, n_frames):
    data = np.random.default_rng(0).random((n_features, n_frames)).astype(np.float32)
    np.savez(str(path), feature=data)
    return data


_CFG_STR = """
[misc]
seed=42
[transforms.annots]
time_precision=0.001
min_label_duration=0.02
lonely_labels=[]
min_silence_gap=0.001
silence_tag="SIL"
[transforms.audio]
audio_features=["mfcc"]
sampling_rate=44100
n_mfcc=13
hop_length=0.01
win_length=0.02
n_fft=2048
fmin=500
fmax=8000
lifter=40
[transforms.audio.delta]
padding="wrap"
[transforms.audio.delta2]
padding="wrap"
[transforms.training]
max_sequences=-1
test_ratio=0.2
[transforms.training.balance]
min_silence_duration=0.2
[transforms.training.balance.data_augmentation]
noise_std=0.01
[model.syn]
units=50
sr=0.4
leak=0.1
iss=0.0005
isd=0.02
isd2=0.002
ridge=1e-8
backend="threading"
workers=1
[model.nsyn]
units=50
sr=0.4
leak=0.1
iss=0.0005
isd=0.02
isd2=0.002
ridge=1e-8
backend="threading"
workers=1
[correction]
min_segment_proportion_agreement=0.66
"""


def _make_corpus_with_mfcc(tmp_path, n_seqs=6):
    """Corpus with syn_mfcc resource and train/test split."""
    from canapy.corpus import Corpus

    config = Config(**toml.loads(_CFG_STR))
    labels = list("ABC")
    rows, mfcc_rows = [], []

    for i in range(n_seqs):
        lbl = labels[i % len(labels)]
        wav = f"seq_{i:02d}.wav"
        npz = tmp_path / f"seq_{i:02d}.mfcc.npz"
        np.savez(str(npz), feature=np.random.default_rng(i).random((13, 100)).astype(np.float32))
        rows.append({
            "label": lbl,
            "onset_s": 0.0,
            "offset_s": 0.5,
            "notated_path": wav,
            "annot_path": wav.replace(".wav", ".csv"),
            "sequence": 0,
            "annotation": wav,
            "train": i < (n_seqs // 2),
        })
        mfcc_rows.append({"notated_path": wav, "feature_path": str(npz)})

    df = pd.DataFrame(rows)
    corpus = Corpus.from_df(df, config=config)
    corpus.dataset = df.copy()
    corpus.register_data_resource("syn_mfcc", pd.DataFrame(mfcc_rows))
    return corpus


# ===========================================================================
# _split_hp_val
# ===========================================================================

class TestSplitHpVal:
    def test_single_seqid_returns_identical_lists(self):
        from canapy.optimization import _split_hp_val
        rng = np.random.default_rng(0)
        df = _make_df_prepared(["s0"], ["A"])
        hp_train, hp_val = _split_hp_val(["s0"], df, 0.2, rng)
        assert set(hp_train) == {"s0"}
        assert set(hp_val) == {"s0"}

    def test_two_seqids_both_lists_non_empty(self):
        from canapy.optimization import _split_hp_val
        rng = np.random.default_rng(0)
        df = _make_df_prepared(["s0", "s1"], ["A", "A"])
        hp_train, hp_val = _split_hp_val(["s0", "s1"], df, 0.5, rng)
        assert len(hp_train) >= 1
        assert len(hp_val) >= 1

    def test_all_classes_represented_in_hp_train(self):
        from canapy.optimization import _split_hp_val
        rng = np.random.default_rng(42)
        seqids = [f"s{i}" for i in range(10)]
        labels = ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"]
        df = _make_df_prepared(seqids, labels)
        hp_train, _ = _split_hp_val(seqids, df, 0.2, rng)
        train_labels = set(df[df["seqid"].isin(hp_train)]["label"])
        assert train_labels == {"A", "B", "C"}

    def test_hp_val_never_empty(self):
        from canapy.optimization import _split_hp_val
        rng = np.random.default_rng(0)
        seqids = [f"s{i}" for i in range(6)]
        labels = ["A", "B"] * 3
        df = _make_df_prepared(seqids, labels)
        _, hp_val = _split_hp_val(seqids, df, 0.2, rng)
        assert len(hp_val) >= 1

    def test_union_covers_all_seqids(self):
        from canapy.optimization import _split_hp_val
        rng = np.random.default_rng(7)
        seqids = [f"s{i}" for i in range(8)]
        labels = ["A", "B"] * 4
        df = _make_df_prepared(seqids, labels)
        hp_train, hp_val = _split_hp_val(seqids, df, 0.25, rng)
        assert set(hp_train) | set(hp_val) == set(seqids)

    def test_val_ratio_in_reasonable_range(self):
        from canapy.optimization import _split_hp_val
        rng = np.random.default_rng(99)
        seqids = [f"s{i}" for i in range(20)]
        labels = ["A", "B"] * 10
        df = _make_df_prepared(seqids, labels)
        hp_train, hp_val = _split_hp_val(seqids, df, 0.3, rng)
        actual_ratio = len(hp_val) / len(seqids)
        assert 0.10 <= actual_ratio <= 0.50

    def test_reproducible_with_same_rng_seed(self):
        from canapy.optimization import _split_hp_val
        seqids = [f"s{i}" for i in range(12)]
        labels = ["A", "B", "C"] * 4
        df = _make_df_prepared(seqids, labels)
        t1, v1 = _split_hp_val(seqids, df, 0.25, np.random.default_rng(123))
        t2, v2 = _split_hp_val(seqids, df, 0.25, np.random.default_rng(123))
        assert set(t1) == set(t2)
        assert set(v1) == set(v2)


# ===========================================================================
# _load_selected_seqids_to_memmap
# ===========================================================================

class TestLoadSelectedSeqidsToMemmap:
    def _setup_single_seq(self, tmp_path, n_features=10, n_frames=50, n_classes=3):
        npz_path = tmp_path / "mfcc.npz"
        _make_npz(npz_path, n_features, n_frames)

        mfcc_paths = pd.DataFrame({
            "notated_path": ["audio.wav"],
            "feature_path": [str(npz_path)],
        })
        enc = np.array([1.0, 0.0, 0.0])
        df_prepared = pd.DataFrame({
            "seqid":        ["seq0", "seq0"],
            "notated_path": ["audio.wav", "audio.wav"],
            "onset_spec":   [0, 25],
            "offset_spec":  [25, 50],
            "encoded_label": [enc, enc],
        })
        frame_estimates = {"seq0": n_frames}
        return mfcc_paths, df_prepared, frame_estimates

    def test_output_shapes(self, tmp_path):
        from canapy.optimization import _load_selected_seqids_to_memmap
        n_feat, n_frames, n_cls = 10, 50, 3
        mfcc_paths, df, frame_est = self._setup_single_seq(tmp_path, n_feat, n_frames, n_cls)
        with tempfile.TemporaryDirectory() as mmap_dir:
            X, Y, lengths = _load_selected_seqids_to_memmap(
                None, ["seq0"], df, mfcc_paths, frame_est, n_cls, mmap_dir, "test"
            )
        assert X.shape == (n_frames, n_feat)
        assert Y.shape == (n_frames, n_cls)

    def test_seq_lengths_sum_equals_total_frames(self, tmp_path):
        from canapy.optimization import _load_selected_seqids_to_memmap
        mfcc_paths, df, frame_est = self._setup_single_seq(tmp_path)
        with tempfile.TemporaryDirectory() as mmap_dir:
            X, _, lengths = _load_selected_seqids_to_memmap(
                None, ["seq0"], df, mfcc_paths, frame_est, 3, mmap_dir, "test"
            )
        assert sum(lengths) == X.shape[0]

    def test_two_sequences_concatenated(self, tmp_path):
        from canapy.optimization import _load_selected_seqids_to_memmap
        n_features, n_classes = 8, 2
        npz0 = tmp_path / "m0.npz"
        npz1 = tmp_path / "m1.npz"
        _make_npz(npz0, n_features, 30)
        _make_npz(npz1, n_features, 40)

        mfcc_paths = pd.DataFrame({
            "notated_path": ["a0.wav", "a1.wav"],
            "feature_path": [str(npz0), str(npz1)],
        })
        enc = np.array([1.0, 0.0])
        df = pd.DataFrame({
            "seqid":        ["s0", "s1"],
            "notated_path": ["a0.wav", "a1.wav"],
            "onset_spec":   [0, 0],
            "offset_spec":  [30, 40],
            "encoded_label": [enc, enc],
        })
        frame_est = {"s0": 30, "s1": 40}

        with tempfile.TemporaryDirectory() as mmap_dir:
            X, Y, lengths = _load_selected_seqids_to_memmap(
                None, ["s0", "s1"], df, mfcc_paths, frame_est, n_classes, mmap_dir, "two"
            )
        assert X.shape[0] == 70
        assert Y.shape[0] == 70
        assert lengths == [30, 40]

    def test_empty_seqids_raises_value_error(self, tmp_path):
        from canapy.optimization import _load_selected_seqids_to_memmap
        mfcc_paths = pd.DataFrame({"notated_path": [], "feature_path": []})
        df = pd.DataFrame({
            "seqid": [], "notated_path": [], "onset_spec": [],
            "offset_spec": [], "encoded_label": [],
        })
        with tempfile.TemporaryDirectory() as mmap_dir:
            with pytest.raises(ValueError):
                _load_selected_seqids_to_memmap(
                    None, [], df, mfcc_paths, {}, 3, mmap_dir, "empty"
                )

    def test_label_one_hot_written_correctly(self, tmp_path):
        from canapy.optimization import _load_selected_seqids_to_memmap
        n_features, n_classes = 5, 3
        npz_path = tmp_path / "m.npz"
        _make_npz(npz_path, n_features, 20)

        mfcc_paths = pd.DataFrame({
            "notated_path": ["a.wav"],
            "feature_path": [str(npz_path)],
        })
        enc_a = np.array([1.0, 0.0, 0.0])
        enc_b = np.array([0.0, 1.0, 0.0])
        df = pd.DataFrame({
            "seqid":        ["s0", "s0"],
            "notated_path": ["a.wav", "a.wav"],
            "onset_spec":   [0, 10],
            "offset_spec":  [10, 20],
            "encoded_label": [enc_a, enc_b],
        })

        with tempfile.TemporaryDirectory() as mmap_dir:
            _, Y, _ = _load_selected_seqids_to_memmap(
                None, ["s0"], df, mfcc_paths, {"s0": 20}, n_classes, mmap_dir, "lbl"
            )

        assert np.all(Y[:10, 0] == 1.0)
        assert np.all(Y[:10, 1] == 0.0)
        assert np.all(Y[10:, 1] == 1.0)
        assert np.all(Y[10:, 0] == 0.0)


# ===========================================================================
# _concat_xy
# ===========================================================================

class TestConcatXY:
    def test_basic_concatenation_shape(self):
        from canapy.optimization import _concat_xy
        X = [np.ones((10, 3)), np.ones((20, 3))]
        Y = [np.zeros((10, 2)), np.zeros((20, 2))]
        xc, yc = _concat_xy(X, Y)
        assert xc.shape == (30, 3)
        assert yc.shape == (30, 2)

    def test_trims_to_min_length_per_sequence(self):
        from canapy.optimization import _concat_xy
        X = [np.ones((10, 3))]
        Y = [np.zeros((15, 2))]
        xc, yc = _concat_xy(X, Y)
        assert xc.shape[0] == 10
        assert yc.shape[0] == 10

    def test_1d_y_gets_reshaped_to_2d(self):
        from canapy.optimization import _concat_xy
        X = [np.ones((5, 3))]
        Y = [np.zeros(5)]
        xc, yc = _concat_xy(X, Y)
        assert yc.ndim == 2
        assert yc.shape == (5, 1)

    def test_values_preserved(self):
        from canapy.optimization import _concat_xy
        X = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        Y = [np.array([[0.0, 1.0], [1.0, 0.0]])]
        xc, yc = _concat_xy(X, Y)
        np.testing.assert_array_equal(xc, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_multiple_sequences_concatenated(self):
        from canapy.optimization import _concat_xy
        rng = np.random.default_rng(0)
        X = [rng.random((l, 4)) for l in [10, 20, 15]]
        Y = [rng.random((l, 3)) for l in [10, 20, 15]]
        xc, yc = _concat_xy(X, Y)
        assert xc.shape[0] == 45
        assert yc.shape[0] == 45


# ===========================================================================
# _arrays_to_memmap
# ===========================================================================

class TestArraysToMemmap:
    def test_creates_one_memmap_per_array(self, tmp_path):
        from canapy.optimization import _arrays_to_memmap
        arrays = [np.arange(10, dtype=np.float32), np.arange(20, dtype=np.float32)]
        result = _arrays_to_memmap(arrays, str(tmp_path))
        assert len(result) == 2

    def test_data_preserved_in_memmap(self, tmp_path):
        from canapy.optimization import _arrays_to_memmap
        arr = np.arange(100, dtype=np.float64)
        result = _arrays_to_memmap([arr], str(tmp_path))
        np.testing.assert_array_equal(result[0], arr)

    def test_dat_files_created_on_disk(self, tmp_path):
        from canapy.optimization import _arrays_to_memmap
        _arrays_to_memmap([np.zeros(10), np.zeros(20)], str(tmp_path))
        dat_files = list(tmp_path.glob("arr_*.dat"))
        assert len(dat_files) == 2

    def test_2d_array_shape_preserved(self, tmp_path):
        from canapy.optimization import _arrays_to_memmap
        arr = np.ones((5, 8), dtype=np.float32)
        result = _arrays_to_memmap([arr], str(tmp_path))
        assert result[0].shape == (5, 8)

    def test_result_is_memmap(self, tmp_path):
        from canapy.optimization import _arrays_to_memmap
        arr = np.ones(10, dtype=np.float32)
        result = _arrays_to_memmap([arr], str(tmp_path))
        assert isinstance(result[0], np.memmap)


# ===========================================================================
# _get_seq_metadata
# ===========================================================================

class TestGetSeqMetadata:
    def test_returns_five_values(self, tmp_path):
        from canapy.optimization import _get_seq_metadata
        corpus = _make_corpus_with_mfcc(tmp_path)
        result = _get_seq_metadata(corpus, purpose="training")
        assert len(result) == 5

    def test_seqids_is_non_empty_list(self, tmp_path):
        from canapy.optimization import _get_seq_metadata
        corpus = _make_corpus_with_mfcc(tmp_path)
        seqids, _, _, _, _ = _get_seq_metadata(corpus, purpose="training")
        assert isinstance(seqids, list)
        assert len(seqids) > 0

    def test_frame_estimates_is_dict(self, tmp_path):
        from canapy.optimization import _get_seq_metadata
        corpus = _make_corpus_with_mfcc(tmp_path)
        _, frame_estimates, _, _, _ = _get_seq_metadata(corpus, purpose="training")
        assert isinstance(frame_estimates, dict)

    def test_class_weights_is_dict(self, tmp_path):
        from canapy.optimization import _get_seq_metadata
        corpus = _make_corpus_with_mfcc(tmp_path)
        _, _, class_weights, _, _ = _get_seq_metadata(corpus, purpose="training")
        assert isinstance(class_weights, dict)

    def test_all_seqids_have_frame_estimate(self, tmp_path):
        from canapy.optimization import _get_seq_metadata
        corpus = _make_corpus_with_mfcc(tmp_path)
        seqids, frame_estimates, _, _, _ = _get_seq_metadata(corpus, purpose="training")
        for s in seqids:
            assert s in frame_estimates

    def test_train_and_eval_seqids_are_disjoint(self, tmp_path):
        from canapy.optimization import _get_seq_metadata
        corpus = _make_corpus_with_mfcc(tmp_path, n_seqs=8)
        seqids_train, _, _, _, _ = _get_seq_metadata(corpus, purpose="training")
        seqids_eval, _, _, _, _ = _get_seq_metadata(corpus, purpose="eval")
        assert set(seqids_train).isdisjoint(set(seqids_eval))


# ===========================================================================
# _select_sequences_by_frames
# ===========================================================================

class TestSelectSequencesByFrames:
    def _make_inputs(self, n=10, frames_each=100):
        seqids = [f"s{i}" for i in range(n)]
        frame_estimates = {s: frames_each for s in seqids}
        class_weights = {s: 1.0 / n for s in seqids}
        return seqids, frame_estimates, class_weights

    def test_returns_list(self):
        from canapy.optimization import _select_sequences_by_frames
        seqids, fe, cw = self._make_inputs()
        result = _select_sequences_by_frames(seqids, fe, cw, max_percentage=0.5, rng=np.random.default_rng(0))
        assert isinstance(result, list)

    def test_max_percentage_one_returns_all(self):
        from canapy.optimization import _select_sequences_by_frames
        seqids, fe, cw = self._make_inputs()
        result = _select_sequences_by_frames(seqids, fe, cw, max_percentage=1.0, rng=np.random.default_rng(0))
        assert set(result) == set(seqids)

    def test_max_percentage_less_than_one_selects_subset(self):
        from canapy.optimization import _select_sequences_by_frames
        seqids, fe, cw = self._make_inputs(n=20)
        result = _select_sequences_by_frames(seqids, fe, cw, max_percentage=0.3, rng=np.random.default_rng(0))
        assert len(result) < len(seqids)

    def test_selected_is_subset_of_original(self):
        from canapy.optimization import _select_sequences_by_frames
        seqids, fe, cw = self._make_inputs()
        result = _select_sequences_by_frames(seqids, fe, cw, max_percentage=0.5, rng=np.random.default_rng(0))
        assert all(s in seqids for s in result)

    def test_empty_seqids_returns_empty(self):
        from canapy.optimization import _select_sequences_by_frames
        result = _select_sequences_by_frames([], {}, {}, max_percentage=1.0, rng=np.random.default_rng(0))
        assert result == []
