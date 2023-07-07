# Author: Nathan Trouvain at 07/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import numpy as np
import pandas as pd


def frames_to_seconds(frame_indices, frame_size, sampling_rate, time_precision=0.001):
    frame_positions = frame_indices * frame_size

    # Frames are usually centered around (frame_size * frame_index) in original
    # audio
    onsets_f = frame_positions - frame_size / 2
    offsets_f = frame_positions + frame_size / 2

    # Correct at edges to remove padding
    onsets_f[0] = 0.0
    offsets_f[-1] = offsets_f[-1] - frame_size / 2

    # Round to the closest sample position
    onsets_f = np.ceil(onsets_f)
    offsets_f = np.floor(offsets_f)

    # Convert to seconds
    decimals = round(-np.log10(time_precision))

    onsets_f = np.around(onsets_f / sampling_rate, decimals=decimals)
    offsets_f = np.around(offsets_f / sampling_rate, decimals=decimals)

    return onsets_f, offsets_f


def seconds_to_frames(timings, frame_size, sampling_rate):
    sample_timings = np.round(timings * sampling_rate).astype(int)
    frame_indices = np.ceil((sample_timings + frame_size // 2) / frame_size)

    return frame_indices.astype(int)


def frames_to_timed_df(
    frame_predictions,
    notated_path,
    frame_size,
    sampling_rate,
    time_precision=0.001,
):
    n_frames = len(frame_predictions)

    if n_frames == 0:
        return pd.DataFrame(columns=["label", "onset_s", "offset_s", "notated_path"])

    frame_indices = np.arange(n_frames)

    onset_s, offset_s = frames_to_seconds(
        frame_indices,
        frame_size=frame_size,
        sampling_rate=sampling_rate,
        time_precision=time_precision,
    )

    df = pd.DataFrame(
        {
            "label": frame_predictions,
            "onset_s": onset_s,
            "offset_s": offset_s,
            "notated_path": [notated_path] * n_frames,
        }
    )

    return df
