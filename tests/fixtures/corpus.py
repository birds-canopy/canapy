# Author: Nathan Trouvain at 05/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import numpy as np
import pandas as pd
import pytest

from canapy.corpus import Corpus


@pytest.fixture()
def corpus():
    c = Corpus.from_directory(
        audio_directory="/home/nathan/Documents/Code/canapy-test/data/",
        annots_directory="/home/nathan/Documents/Code/canapy-test/data/",
        )
    return c


@pytest.fixture()
def df():
    df = pd.DataFrame(
        {
            "label": list("abcdefghabcdefgh"),
            "onset_s": np.linspace(0.0, 0.7, 8).tolist() + np.linspace(0.1, 0.8, 8).tolist(),
            "offset_s": np.linspace(0.1, 0.8, 8).tolist() + np.linspace(0.2, 0.9, 8).tolist(),
            "notated_path": ["foo.wav"] * 8 + ["baz.wav"] * 8,
            "annots_path": ["foo.csv"] * 8 + ["baz.csv"] * 8,
            }
        )

    return df


@pytest.fixture()
def predictions_df():
    df = pd.DataFrame(
        {
            "label": list("abcdefghabcdefgh"),
            "onset_s": np.linspace(0.0, 0.7, 8).tolist() + np.linspace(0.1, 0.8, 8).tolist(),
            "offset_s": np.linspace(0.1, 0.8, 8).tolist() + np.linspace(0.2, 0.9, 8).tolist(),
            "notated_path": ["foo.wav"] * 8 + ["baz.wav"] * 8,
            }
        )

    return df
