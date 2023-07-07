"""Tests of canapy.dataset module, using pytest"""
import json

from pathlib import Path

import pytest
import pandas as pd

from canapy import Dataset, Config


@pytest.fixture(scope="session")
def test_data():
    """Path to test data"""
    return Path("./data/test-rouge6/")


@pytest.fixture(scope="session")
def audio_data():
    """Path to audio files and labels csv"""
    return Path("./data/test-rouge6/test-raw")


@pytest.fixture(scope="session")
def audio_files():
    """Path to audio files and labels csv"""
    return [
        f for f in Path("./data/test-rouge6/test-raw").iterdir() if f.suffix == ".wav"
    ]


@pytest.fixture(scope="session")
def corrections():
    """Correction file as dict"""
    with Path("./data/test-rouge6/test.corrections.json").open("r") as f:
        corr = json.load(f)
    return corr


@pytest.fixture(scope="session")
def config_dict():
    """Config file as dict"""
    with Path("./data/test-rouge6/test.config.json").open("r") as f:
        conf = json.load(f)
    return conf


@pytest.fixture(scope="session")
def config():
    """Config file as Config object"""
    with Path("./data/test-rouge6/test.config.json").open("r") as f:
        conf = Config(json.load(f))
    return conf


@pytest.fixture(scope="session")
def dataset():
    """Dataset object"""
    return Dataset("./data/test-rouge6/")


@pytest.fixture
def dummy_df():
    return pd.DataFrame(
        {
            "wave": ["bar"] * 5 + ["foo"] * 3,
            "start": [1.21, 1.5, 2.35, 3.35, 3.43, 1.26, 1.69, 4.02],
            "end": [1.211, 2.3, 2.41, 3.42, 3.58, 1.58, 4.001, 4.12],
            "syll": ["cri", "cri", "cri", "A", "B", "C123", "U", "Y"],
        }
    )


@pytest.fixture
def dummy_correction():
    return {"syll": {"A": "B"}, "sample": {5: "C12"}}


### TESTS ###


def test_new_dataset(test_data, audio_data, audio_files, config, corrections):
    dataset = Dataset(test_data)

    assert dataset.directory == test_data
    assert dataset.audiodir == audio_data
    assert dataset.audios == audio_files
    assert len(dataset.vocab) > 0
    assert list(dataset.config.keys()) == list(config.keys())
    assert list(dataset.config.values()) == list(config.values())
    assert dataset.corrections == corrections


def test_config_get(config):
    assert config.get("n_mfcc") == 20
    assert config.get("n_mfcc", "lifter") == (20, 40)
    assert config["n_mfcc"] == 20
    assert config.n_mfcc == 20
    assert type(config.syn) is Config
    assert config.syn.N == 1000


def test_config_utils(config):
    assert config.as_frames("n_fft") == 1102
    assert round(config.as_duration("sampling_rate")) == 1
    assert config.as_fftwindow("n_fft") == 1024
    assert round(config.duration(44100)) == 1
    assert config.steps(1.0) == 44100
    assert config.frames(1.0) == 86


def test_split(dataset, audio_files):
    train_syn, test_syn = dataset.split_syn()
    assert 0.10 < len(test_syn.groupby("wave")) / len(train_syn.groupby("wave")) < 0.30
    assert len(train_syn.groupby("wave")) + len(test_syn.groupby("wave")) == len(
        audio_files
    )

    train_syn, test_syn = dataset.split_syn(max_songs=30)
    assert len(train_syn.groupby("wave")) == 30

    train_syn, test_syn = dataset.split_syn(max_songs=5)
    assert len(train_syn.groupby("wave")) == 5


def test_balance(dataset):
    df = dataset.balance_nonsyn()
    df["duration"] = df["end"] - df["start"]
    assert 40 > df.groupby("syll")["duration"].sum().min() > 29


def test_trainset(dataset, audio_files):
    songs, phrases, test = dataset.to_trainset()

    assert len(songs.groupby("wave")) + len(test.groupby("wave")) == len(audio_files)
    assert sorted(list(phrases.groupby("wave").groups.keys())) == sorted(
        list(songs.groupby("wave").groups.keys())
    )
    assert not (
        sorted(list(songs.groupby("wave").groups.keys()))
        == sorted(list(test.groupby("wave").groups.keys()))
    )
    assert not (
        sorted(list(phrases.groupby("wave").groups.keys()))
        == sorted(list(test.groupby("wave").groups.keys()))
    )

    songs, phrases, test = dataset.to_trainset(max_songs=30)

    assert len(songs.groupby("wave")) == 30
    assert sorted(list(phrases.groupby("wave").groups.keys())) == sorted(
        list(songs.groupby("wave").groups.keys())
    )
    assert not (
        sorted(list(songs.groupby("wave").groups.keys()))
        == sorted(list(test.groupby("wave").groups.keys()))
    )
    assert not (
        sorted(list(phrases.groupby("wave").groups.keys()))
        == sorted(list(test.groupby("wave").groups.keys()))
    )


def test_refine(dataset, dummy_df, dummy_correction):
    df = dataset._apply_corrections(df=dummy_df, corrections=dummy_correction)
    assert df.syll.tolist() == ["cri", "cri", "cri", "B", "B", "C12", "U", "Y"]

    df = dataset._remove_short_samples(df=df)
    assert df.syll.tolist() == ["cri", "cri", "B", "B", "C12", "U", "Y"]

    df = dataset._join_groups(df)
    assert df.syll.tolist() == ["cri", "cri", "B", "C12", "U", "Y"]

    assert df.index.tolist() == [0, 1, 2, 3, 4, 5]

    df = dataset._tag_silence(df)
    assert (
        len(
            {*df.syll.tolist()}
            - {"cri", "SIL", "cri", "SIL", "B", "SIL", "C12", "SIL", "U", "SIL", "Y"}
        )
        == 0
    )


@pytest.mark.parametrize(
    "split,max_songs,mode",
    [(True, None, None), (False, None, "syn"), (True, 30, None), (False, 30, "syn")],
)
def test_to_features(dataset, split, max_songs, mode):
    if split:
        (syn, ysyn), (test, ytest) = dataset.to_features(
            split=split, max_songs=max_songs, mode=mode
        )

        if not max_songs:
            assert len(test) + len(syn) == len(dataset.df.groupby("wave"))
        else:
            assert len(syn) == max_songs
        assert test[0].shape[1] == 60
        assert ytest[0].shape == (test[0].shape[0], len(dataset.vocab))
    else:
        (syn, ysyn) = dataset.to_features(split=split, mode=mode, max_songs=max_songs)

        if max_songs:
            assert len(syn) == max_songs
        else:
            assert len(syn) == len(dataset.df.groupby("wave"))

        assert len(syn) == len(ysyn)
        assert syn[0].shape[1] == 60
        assert ysyn[0].shape == (syn[0].shape[0], len(dataset.vocab))


@pytest.mark.parametrize(
    "old,new,expected",
    [
        (
            {
                "syll": {"A1": "A", "A2": "A", "B": "X", "E": "Z"},
                "sample": {10: "C", 11: "A", 12: "X"},
            },
            {"syll": {"A": "foo", "X": "bar"}, "sample": {13: "V"}},
            {
                "syll": {
                    "A1": "foo",
                    "A2": "foo",
                    "A": "foo",
                    "B": "bar",
                    "X": "bar",
                    "E": "Z",
                },
                "sample": {10: "C", 11: "foo", 12: "bar", 13: "V"},
            },
        ),
        (
            {
                "syll": {"A1": "A", "A": "B", "B": "X", "E": "Z"},
                "sample": {10: "C", 11: "A", 12: "X"},
            },
            {"syll": {"B": "foo", "X": "bar"}, "sample": {13: "V"}},
            {
                "syll": {"A1": "foo", "A": "foo", "B": "foo", "X": "bar", "E": "Z"},
                "sample": {10: "C", 11: "foo", 12: "bar", 13: "V"},
            },
        ),
    ],
)
def test_update_corrections(dataset, old, new, expected):
    dataset.corrections = old
    updated = dataset.update_corrections(new)

    assert updated == expected


def test_switch(dataset):
    old_vocab = dataset.vocab.copy()
    dataset.switch("./data/test-rouge6-noannots")

    assert dataset.df.get("syll") is None and dataset.df.get("wave") is not None
    assert len(dataset.df) == len(dataset.audios)
    assert dataset.audiodir == Path("./data/test-rouge6-noannots")
    assert dataset.vocab == old_vocab
