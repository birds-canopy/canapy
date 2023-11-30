# Author: Nathan Trouvain at 27/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import warnings

import pandas as pd
import numpy as np

from ...log import log


@log(fn_type="corpus transform")
def sort_annotations(corpus, **kwargs):
    df = corpus.dataset.sort_values(
        by=["annotation", "sequence", "onset_s"],
        ascending=True,
        ignore_index=True,
    )

    return corpus.clone_with_df(df)


@log(fn_type="corpus transform")
def tag_silences(corpus, **kwargs):
    df = corpus.dataset

    sil_start = df.groupby(["notated_path"])["offset_s"].shift(fill_value=0.0)
    sil_end = df["onset_s"]

    silence_df = df.copy(deep=True)

    silence_df["onset_s"] = sil_start
    silence_df["offset_s"] = sil_end
    silence_df["label"] = corpus.config.transforms.annots.silence_tag

    too_short = silence_df[sil_end - sil_start <= 0.0].index
    silence_df.drop(too_short, axis=0, inplace=True)

    df = pd.concat([df, silence_df], ignore_index=True)

    return corpus.clone_with_df(df)


@log(fn_type="corpus transform")
def remove_short_labels(corpus, **kwargs):
    df = corpus.dataset
    config = corpus.config.transforms.annots

    durations = df["offset_s"] - df["onset_s"]
    durations.round(decimals=round(-np.log10(config.time_precision)))

    too_short = df[durations < config.min_label_duration].index

    df = df.drop(too_short, axis=0).reset_index()

    return corpus.clone_with_df(df)


@log(fn_type="corpus transform")
def merge_labels(corpus, **kwargs):
    df = corpus.dataset
    config = corpus.config.transforms.annots

    groups = ["annotation", "sequence"]

    gemini_groups = (
        (df["label"].shift(fill_value=str(np.nan)) != df["label"])
        & ~(df["label"].isin(config.lonely_labels))
        & (
            df["onset_s"].shift(fill_value=0.0) - df["offset_s"]
            <= config.min_silence_gap
        )
    ).cumsum()

    # We define aggregation functions this way so that we don't inadvertedly
    # remove unknown columns from the dataset.
    agg_funcs = {}
    for col in df.columns:
        if "onset" in col:
            agg_funcs[col] = "min"
        elif "offset" in col:
            agg_funcs[col] = "max"
        else:
            agg_funcs[col] = "first"
    
    # Remove pandas FutureWarning with groupby and as_index=False.
    # Grouped columns must be preserved here and not turned into index.
    with warnings.catch_warnings(): 
        warnings.simplefilter("ignore")
        df = (
            df.groupby(groups + [gemini_groups], as_index=False)
            .agg(agg_funcs)
            .sort_values(by=["annotation", "sequence", "onset_s"])
        )

    return corpus.clone_with_df(df)
