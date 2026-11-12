# Licence: BSD-3-Clause
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from canapy.corpus import Corpus
from canapy.outliers import find_outliers, find_shifted_files, outliers_params

SR = 44100
TONES = {"A": 1000.0, "B": 2000.0, "C": 3500.0, "D": 5500.0}  # Hz, one tone per label
SEGMENT_S = 0.2
GAP_S = 0.1
N_FILES = 30
N_SEGMENTS = 8  # per file
MISLABELED = [(1, 2, "D"), (5, 4, "B"), (9, 6, "C")]  # (file, segment, wrong label)
SHIFTED_FILE, SHIFT_S = 3, SEGMENT_S + GAP_S  # its annotations one segment late


def write_songs(directory, mislabeled=(), shifted=None):
    """Songs of tone segments separated by silences, and their annotations."""
    rng = np.random.default_rng(0)
    rows = []
    for f in range(N_FILES):
        labels = [rng.choice(list(TONES))]
        while len(labels) < N_SEGMENTS:  # (consecutive segments of different labels)
            labels.append(rng.choice([label for label in TONES if label != labels[-1]]))
        chunks, cursor = [rng.normal(0, 1e-3, int(GAP_S * SR))], GAP_S
        for k, label in enumerate(labels):
            t = np.arange(int(SEGMENT_S * SR)) / SR
            chunks.append(np.sin(2 * np.pi * TONES[label] * t) + rng.normal(0, 1e-2, len(t)))
            chunks.append(rng.normal(0, 1e-3, int(GAP_S * SR)))
            shift = SHIFT_S if f == shifted else 0.0
            rows.append({"label": label, "onset_s": cursor + shift,
                         "offset_s": cursor + SEGMENT_S + shift, "file": f, "segment": k})
            cursor += SEGMENT_S + GAP_S
        path = directory / f"song_{f:02d}.wav"
        sf.write(str(path), np.concatenate(chunks + [rng.normal(0, 1e-3, int(SR))]), SR)
        for row in rows[-N_SEGMENTS:]:
            row["notated_path"] = str(path)
    df = pd.DataFrame(rows)
    df["tone"] = df["label"]
    for f, k, label in mislabeled:
        df.loc[(df.file == f) & (df.segment == k), "label"] = label
    df["annot_path"] = [str(directory / f"song_{f:02d}.csv") for f in df["file"]]
    return df.drop(columns=["file", "segment"])


def make_corpus(directory, config, **kwargs):
    """(corpus of the songs, the label of the tone of each segment by its (file, onset))"""
    df = write_songs(directory, **kwargs)
    corpus = Corpus.from_df(df.drop(columns="tone"), config=config)
    corpus.spec_directory = directory
    corpus.spec_ext = ".mfcc.npz"
    return corpus, dict(zip(zip(df["notated_path"], df["onset_s"]), df["tone"]))


@pytest.fixture()
def params(config):
    # (99 % of 240 segments: the two furthest ones; 95 % leaves the 3 mislabeled ones)
    return {**outliers_params(config), "percentile": 95.0}


def test_mislabeled_segments_are_found(tmp_path, config, params):
    corpus, tones = make_corpus(tmp_path, config, mislabeled=MISLABELED)
    result = find_outliers(corpus, params, seed=0)

    assert len(result) == N_FILES * N_SEGMENTS
    assert {"umap_0", "umap_1", "umap_2"} <= set(result.columns)
    found = result[result["outlier"]]
    expected = {(f"song_{f:02d}.wav", label) for f, _, label in MISLABELED}
    assert {(p.split("/")[-1], label) for p, label in zip(found["notated_path"], found["label"])} == expected
    # they sound like the label of their tone
    assert list(found["sounds_like"]) == [tones[p, t] for p, t in zip(found["notated_path"], found["onset_s"])]


def test_clean_corpus_has_no_outlier(tmp_path, config, params):
    result = find_outliers(make_corpus(tmp_path, config)[0], params, seed=0)
    assert not result["outlier"].any()
    assert (result["sounds_like"] == result["label"]).all()


def test_keys_match_the_dataset(tmp_path, config, params):
    from canapy.correction import match_annotation

    corpus, _ = make_corpus(tmp_path, config, mislabeled=MISLABELED)
    result = find_outliers(corpus, params, seed=0)
    for key in result.loc[result["outlier"], "key"]:
        assert len(match_annotation(corpus.dataset, key)) == 1


def test_shifted_file_is_found_with_its_shift(tmp_path, config, params):
    corpus, _ = make_corpus(tmp_path, config, shifted=SHIFTED_FILE)
    result = find_outliers(corpus, params, seed=0)
    shifted = find_shifted_files(corpus, result, params)

    assert list(shifted["notated_path"].map(lambda p: p.split("/")[-1])) == [f"song_{SHIFTED_FILE:02d}.wav"]
    row = shifted.iloc[0]
    assert row["best_shift"] == pytest.approx(-SHIFT_S, abs=params["shift_step"])
    assert row["knn_agreement_after_shift"] > 0.9 > row["knn_agreement"]


def test_parameters_default_to_the_module(config):
    params = outliers_params(config)  # (the test config has no [outliers] section)
    assert params["neighbors"] == 10 and params["n_components"] == 3
