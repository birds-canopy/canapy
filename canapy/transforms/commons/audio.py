# Author: Nathan Trouvain at 28/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import pathlib

import numpy as np
import librosa as lbr
import pandas as pd

from ...log import log


@log(fn_type="audio transform")
def compute_mfcc(corpus, *, output_directory, resource_name, redo=False, **kwargs):
    df = corpus.dataset
    config = corpus.config.transforms.audio

    if resource_name in corpus.data_resources and not redo:
        return

    audio_paths = df["notated_path"].unique()

    cepstra_paths = []
    for audio_path in audio_paths:
        audio_path = pathlib.Path(audio_path)

        if audio_path.suffix == ".npy":
            audio = np.load(str(audio_path))
            rate = corpus.config.sampling_rate
        else:
            audio, rate = lbr.load(audio_path, sr=config.sampling_rate)

        cepstrum = lbr.feature.mfcc(y=audio, sr=rate, **config.mfcc)

        cepstral_features = []
        if "mfcc" in config.audio_features:
            cepstral_features.append(cepstrum)
        if "delta" in config.audio_features:
            d = lbr.feature.delta(cepstrum, mode=config.delta.padding)
            cepstral_features.append(d)
        if "delta2" in config.audio_features:
            d2 = lbr.feature.delta(cepstrum, order=2, mode=config.delta2.padding)
            cepstral_features.append(d2)

        cepstrum = np.vstack(cepstral_features)

        notated_name = audio_path.stem

        mfcc_path = pathlib.Path(output_directory) / (notated_name + ".mfcc.npy")

        np.save(str(mfcc_path), cepstrum)

        cepstra_paths.append(
            {"notated_path": audio_path.name, "feature_path": mfcc_path}
        )

    cepstrum_df = pd.DataFrame(cepstra_paths)

    corpus.register_data_resource(resource_name, cepstrum_df)

    return corpus
