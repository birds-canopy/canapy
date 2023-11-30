# Author: Nathan Trouvain at 06/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import pytest

import numpy as np
import pandas as pd

from canapy.annotator.commons.postprocess import (
    frames_to_timed_df,
    frames_to_seconds,
    frame_df_to_annots_df,
)


def test_frames_to_timed_df(cls_frame_predictions):
    df = frames_to_timed_df(
        cls_frame_predictions,
        notated_path="foo/baz",
        frame_size=256,
        sampling_rate=44100,
    )
    print(df)


def test_frame_df_to_annots_df(cls_frame_predictions):
    df = frames_to_timed_df(
        cls_frame_predictions,
        notated_path="foo/baz",
        frame_size=100,
        sampling_rate=10000,
    )
    print(df)

    df = frame_df_to_annots_df(df, min_label_duration=0.025, min_silence_gap=0.001)
    print(df)


def test_frame_to_seconds():
    frames_indices = np.arange(300)
    print(frames_to_seconds(frames_indices, 256, 44100))
