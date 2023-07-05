"""Test of canapy.processor module using pytest"""
import pytest

import librosa as lbr
import pandas as pd

from canapy import Dataset
from canapy import Processor


### FIXTURES ###


@pytest.fixture(scope="session")
def dataset():
    """Dataset object"""
    return Dataset("./data/test-rouge6/")


@pytest.fixture(scope="session")
def processor():
    """Processor object"""
    return Processor(Dataset("./data/test-rouge6/"))


@pytest.fixture(scope="session")
def song_samples():
    y1, _ = lbr.load(
        "./data/test-rouge6/test-raw/000-rouge6_July_01_2014_42678325.wav", sr=44100
    )
    y2, _ = lbr.load(
        "./data/test-rouge6/test-raw/001-rouge6_July_01_2014_42678325.wav", sr=44100
    )
    y3, _ = lbr.load(
        "./data/test-rouge6/test-raw/002-rouge6_July_01_2014_42799448.wav", sr=44100
    )

    df = Dataset("./data/test-rouge6/").df

    df1 = df[df["wave"] == "000-rouge6_July_01_2014_42678325.wav"]
    df2 = df[df["wave"] == "001-rouge6_July_01_2014_42678325.wav"]
    df3 = df[df["wave"] == "002-rouge6_July_01_2014_42799448.wav"]

    return [(y1, df1), (y2, df2), (y3, df3)]


@pytest.fixture(scope="session")
def song_samples_noannots():
    y1, _ = lbr.load(
        "./data/test-rouge6/test-raw/000-rouge6_July_01_2014_42678325.wav", sr=44100
    )
    y2, _ = lbr.load(
        "./data/test-rouge6/test-raw/001-rouge6_July_01_2014_42678325.wav", sr=44100
    )
    y3, _ = lbr.load(
        "./data/test-rouge6/test-raw/002-rouge6_July_01_2014_42799448.wav", sr=44100
    )

    df = Dataset("./data/test-rouge6/").df

    df1 = df[df["wave"] == "000-rouge6_July_01_2014_42678325.wav"].drop("syll", axis=1)
    df2 = df[df["wave"] == "001-rouge6_July_01_2014_42678325.wav"].drop("syll", axis=1)
    df3 = df[df["wave"] == "002-rouge6_July_01_2014_42799448.wav"].drop("syll", axis=1)

    return [(y1, df1), (y2, df2), (y3, df3)]


@pytest.fixture(scope="session")
def phrase_samples():
    y1, _ = lbr.load(
        "./data/test-rouge6/test-raw/000-rouge6_July_01_2014_42678325.wav", sr=44100
    )

    dataset = Dataset("./data/test-rouge6/")
    df = dataset.df
    df1 = df[df["wave"] == "000-rouge6_July_01_2014_42678325.wav"]
    df1["augmented"] = True

    phrases = []
    for entry in df1.itertuples():
        start, end = dataset.config.steps(entry.start), dataset.config.steps(entry.end)
        phrases.append((y1[start:end], entry))

    return phrases


### TESTS ###


def test_new_processor(dataset):
    processor = Processor(dataset)

    assert processor.config == dataset.config


@pytest.mark.parametrize(
    "annots",
    [
        pd.DataFrame(
            {
                "wave": ["foo"] * 5,
                "start": [0.0, 1.212, 2.35, 2.41, 3.42],
                "end": [1.211, 2.3, 2.41, 3.42, 3.58],
                "syll": ["cri", "SIL", "cri", "SIL", "B"],
            }
        ),
        pd.DataFrame(
            {"wave": ["foo"], "start": [0.0], "end": [1.211], "syll": ["cri"]}
        ),
    ],
)
def test_tile_teachers(processor, annots):
    y = processor._tile_teachers(annots)
    assert len(y) > 0
    assert y[0] == "cri"
    assert (y == "").sum() == 0


def test_preprocess_song(processor, song_samples):
    for (
        song,
        annots,
    ) in song_samples:
        feats, teachers = processor._preprocess_audio(song, annots)
        assert feats.shape[1] == 60
        assert teachers.shape == (feats.shape[0], len(processor.dataset.vocab))


def test_preprocess_phrase(processor, phrase_samples):
    for (
        song,
        annots,
    ) in phrase_samples:
        feats, teachers = processor._preprocess_audio(song, annots)
        assert feats.shape[1] == 60
        assert teachers.shape == (feats.shape[0], len(processor.dataset.vocab))


def test_preprocess_song_noannots(processor, song_samples_noannots):
    for (
        song,
        annots,
    ) in song_samples_noannots:
        feats, teachers = processor._preprocess_audio(song, annots)
        assert feats.shape[1] == 60
        assert teachers is None
