# Author: Nathan Trouvain at 06/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import pytest
import numpy as np
import pandas as pd


@pytest.fixture()
def cls_frame_predictions():
    return np.array(
        list("bbaaaaaabbaaaabbbbbbbbbbbccccccdddddffffffffefefeeeeeeeeffffffkk")
    )


@pytest.fixture()
def df_frame_predictions():
    frames = np.array(
        list("aaaaaaaaaaabbbbbbbbbbbccccccdddddffffffffefefeeeeeeeeffffff")
    )

    n_frames = len(frames)

    onsets = np.arange(0, n_frames)
    offsets = onsets + 1

    return pd.DataFrame(
        {
            "onset_frame": onsets,
            "offset_frame": offsets,
            "label": frames,
            "notated_path": ["foo/bar"] * n_frames,
        }
    )
