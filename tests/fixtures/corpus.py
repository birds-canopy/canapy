# Author: Nathan Trouvain at 05/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import numpy as np
import pandas as pd
import pytest

from canapy.corpus import Corpus


@pytest.fixture(scope="session")
def corpus():
    c = Corpus.from_directory(
        audio_directory="/home/nathan/Documents/Code/canapy-test/data/",
        annots_directory="/home/nathan/Documents/Code/canapy-test/data/",
    )
    return c


@pytest.fixture(scope="session")
def df():
    df = pd.DataFrame(
        {
            "label": list("abcdefghabcdefgh"),
            "onset_s": np.linspace(0.0, 0.7, 8).tolist()
            + np.linspace(0.1, 0.8, 8).tolist(),
            "offset_s": np.linspace(0.1, 0.8, 8).tolist()
            + np.linspace(0.2, 0.9, 8).tolist(),
            "notated_path": ["foo.wav"] * 8 + ["baz.wav"] * 8,
            "annots_path": ["foo.csv"] * 8 + ["baz.csv"] * 8,
        }
    )

    return df


@pytest.fixture(scope="session")
def predictions_df():
    df = pd.DataFrame(
        {
            "label": list("abcdefghabcdefgh"),
            "onset_s": np.linspace(0.0, 0.7, 8).tolist()
            + np.linspace(0.1, 0.8, 8).tolist(),
            "offset_s": np.linspace(0.1, 0.8, 8).tolist()
            + np.linspace(0.2, 0.9, 8).tolist(),
            "notated_path": ["foo.wav"] * 8 + ["baz.wav"] * 8,
        }
    )

    return df


@pytest.fixture(scope="session")
def prediction_corpus(config):
    classes = list("abcdefghi")

    df = pd.DataFrame(
        {
            "label": list("abcdefghabcdefgh"),
            "onset_s": np.linspace(0.0, 0.7, 8).tolist()
            + np.linspace(0.1, 0.8, 8).tolist(),
            "offset_s": np.linspace(0.1, 0.8, 8).tolist()
            + np.linspace(0.2, 0.9, 8).tolist(),
            "notated_path": ["foo.wav"] * 8 + ["baz.wav"] * 8,
        }
    )

    frame_preds = pd.DataFrame(
        {
            "label": list("aaabbbcccdddeeefffggghhhaaabbbcccdddeeefffggghhh"),
            "onset_s": np.linspace(0.0, 0.7, 8 * 3).tolist()
            + np.linspace(0.1, 0.8, 8 * 3).tolist(),
            "offset_s": np.linspace(0.1, 0.8, 8 * 3).tolist()
            + np.linspace(0.2, 0.9, 8 * 3).tolist(),
            "notated_path": ["foo.wav"] * 8 * 3 + ["baz.wav"] * 8 * 3,
        }
    )

    raw_preds = {
        "foo.wav": np.random.uniform(size=(8 * 3, 8)),
        "baz.wav": np.random.uniform(size=(8 * 3, 8)),
    }

    corpus = Corpus.from_df(df, config=config)

    corpus.register_data_resource("frames_predictions", frame_preds)

    corpus.register_data_resource("nn_output", raw_preds)

    return corpus


@pytest.fixture(scope="session")
def prediction_corpus2(config):
    df = pd.DataFrame(
        {
            "label": list("abcdefghabcdefgh"),
            "onset_s": np.linspace(0.0, 0.7, 8).tolist()
            + np.linspace(0.1, 0.8, 8).tolist(),
            "offset_s": np.linspace(0.1, 0.8, 8).tolist()
            + np.linspace(0.2, 0.9, 8).tolist(),
            "notated_path": ["foo.wav"] * 8 + ["baz.wav"] * 8,
        }
    )

    frame_preds = pd.DataFrame(
        {
            "label": list("aaabbbcccdddeeefffggghhhaaabbbcccdddeeefffggghhh"),
            "onset_s": np.linspace(0.0, 0.7, 8 * 3).tolist()
            + np.linspace(0.1, 0.8, 8 * 3).tolist(),
            "offset_s": np.linspace(0.1, 0.8, 8 * 3).tolist()
            + np.linspace(0.2, 0.9, 8 * 3).tolist(),
            "notated_path": ["foo.wav"] * 8 * 3 + ["baz.wav"] * 8 * 3,
        }
    )


    raw_preds = {
        "foo.wav": np.random.uniform(size=(8 * 3, 8)),
        "baz.wav": np.random.uniform(size=(8 * 3, 8)),
    }

    corpus = Corpus.from_df(df, config=config)

    corpus.register_data_resource("frames_predictions", frame_preds)

    corpus.register_data_resource("nn_output", raw_preds)

    return corpus


@pytest.fixture(scope="session")
def prediction_corpus_no_raw(config):
    df = pd.DataFrame(
        {
            "label": list("abcdefghabcdefgh"),
            "onset_s": np.linspace(0.0, 0.7, 8).tolist()
            + np.linspace(0.1, 0.8, 8).tolist(),
            "offset_s": np.linspace(0.1, 0.8, 8).tolist()
            + np.linspace(0.2, 0.9, 8).tolist(),
            "notated_path": ["foo.wav"] * 8 + ["baz.wav"] * 8,
        }
    )

    frame_preds = pd.DataFrame(
        {
            "label": list("aaabbbcccdddeeefffggghhhaaabbbcccdddeeefffggghhh"),
            "onset_s": np.linspace(0.0, 0.7, 8 * 3).tolist()
            + np.linspace(0.1, 0.8, 8 * 3).tolist(),
            "offset_s": np.linspace(0.1, 0.8, 8 * 3).tolist()
            + np.linspace(0.2, 0.9, 8 * 3).tolist(),
            "notated_path": ["foo.wav"] * 8 * 3 + ["baz.wav"] * 8 * 3,
        }
    )

    corpus = Corpus.from_df(df, config=config)

    corpus.register_data_resource("frames_predictions", frame_preds)

    return corpus
