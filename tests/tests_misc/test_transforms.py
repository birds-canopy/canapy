# Author: Nathan Trouvain at 29/06/2023 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import crowsetta
import numpy as np

from canapy.corpus import Corpus
from canapy.transforms.commons.training import encode_labels
from canapy.transforms.nsynesn import *
from canapy.transforms.synesn import *


def test_balance_labels(corpus):
    df = corpus.dataset
    df["train"] = False
    samples = df.sample(50).index
    df.loc[samples, "train"] = True

    c = balance_labels_duration(corpus, resource_name="balanced_df")

    bdf = c.data_resources["balanced_df"]
    bdf["duration"] = bdf["offset_s"] - bdf["onset_s"]
    durations = bdf.groupby("label")["duration"].sum()

    # All classes should reach at least the median total duration of the original set
    original_durations = df.copy()
    original_durations["duration"] = original_durations["offset_s"] - original_durations["onset_s"]
    expected_min = original_durations.groupby("label")["duration"].sum().median()
    assert durations.min() >= expected_min


def test_compute_mfcc_for_balanced_dataset(corpus):
    df = corpus.dataset
    df["train"] = False
    samples = df.sample(50).index
    df.loc[samples, "train"] = True

    df["augmented"] = False

    # c = balance_labels_duration(corpus, resource_name="balanced_df")

    corpus.register_data_resource("balanced_df", df)

    bdf = c.data_resources["balanced_df"]
    bdf["duration"] = bdf["offset_s"] - bdf["onset_s"]
    durations = bdf.groupby("label")["duration"].sum()

    corpus = compute_mfcc_for_balanced_dataset(corpus, resource_name="mfccs")

    print(corpus)


def test_encode_labels():
    import numpy as np

    # crowsetta.register_format(Marron1CSV)
    c = Corpus.from_directory(
        audio_directory="/home/nathan/Documents/Code/canapy-test/data/",
        annots_directory="/home/nathan/Documents/Code/canapy-test/data/",
    )

    df = c.dataset

    df["train"] = False

    samples = df.sample(50).index
    df.loc[samples, "train"] = True

    print(df.query("train == True"))

    c = encode_labels(c, resource_name="encoded_labels")

    print(c)


if __name__ == "__main__":
    test_encode_labels()
