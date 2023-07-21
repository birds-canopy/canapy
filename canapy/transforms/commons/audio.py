# Author: Nathan Trouvain at 28/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import logging
import pathlib

import numpy as np
import librosa as lbr
import pandas as pd

from ...timings import seconds_to_audio
from ...log import log


logger = logging.getLogger("canapy")


class AudioNotFound(Exception):
    pass


def ls_audio_dir(corpus):
    audio_dir = pathlib.Path(corpus.audio_directory)
    audio_ext = corpus.audio_ext

    audio_paths = list(audio_dir.rglob(f"**/*{audio_ext}"))

    if len(audio_paths) == 0:
        raise AudioNotFound(
            f"No audio data file with extension '{audio_ext}' found in {audio_dir}."
        )

    return audio_paths


def ls_spec_dir(corpus):
    spec_dir = pathlib.Path(corpus.spec_directory)
    spec_ext = corpus.spec_ext

    spec_paths = list(spec_dir.rglob(f"**/*{spec_ext}"))

    if len(spec_paths) == 0:
        return pd.DataFrame(columns=["notated_path", "feature_path"])

    no_audio_file = []
    spec_registry = []
    for spec_path in spec_paths:
        spec = np.load(str(spec_path))

        if spec.dtype.fields is not None and "notated_path" in spec.dtype.fields:
            notated_path = spec["notated_path"][0]
            spec_registry.append(
                {"notated_path": notated_path, "feature_path": spec_path}
            )
        else:
            no_audio_file.append(spec_path)
            spec_registry.append({"notated_path": np.nan, "feature_path": spec_path})

    if len(no_audio_file) > 0:
        logger.warning(
            f"Found {len(no_audio_file)} spectro or feature files with no "
            f"corresponding audio file. If this is not the expected "
            f"behavior and you are providing spectrograms or features as "
            f"Numpy arrays, without attached annotations,you may add audio "
            f"file name to spectrogrames arrays using Numpy structured "
            f"arrays (https://numpy.org/doc/stable/user/basics.rec.html). "
            f"Simply add a field 'notated_path' storing the name of the "
            f"corresponding audio file to the array storing the spectrogram "
            f"and store the spectrogram under the field 'feature'."
        )

    return pd.DataFrame(spec_registry)


@log(fn_type="audio transform")
def compute_mfcc(corpus, *, output_directory, resource_name, redo=False, **kwargs):
    df = corpus.dataset
    config = corpus.config.transforms.audio

    spec_path = corpus.spec_directory
    if spec_path is not None and not redo:
        # look for saved MFCCs from previous computations
        cepstrum_df = ls_spec_dir(corpus)

        # Maybe we have switched corpus but not output directory,
        # and MFCCs must be updated
        if len(set(df["notated_path"].unique()) - set(cepstrum_df["notated_path"].unique())) == 0:
            logger.info(f"Found previously computed MFCCs in {output_directory}. "
                        f"Will use them.")
            corpus.register_data_resource(resource_name, cepstrum_df)
            return corpus
        else:
            logger.warning(f"Mismatch between saved MFCCs in {output_directory} "
                           f"and current audio files in {corpus}. MFCCs will be "
                           f"recomputed.")

    if len(df) > 0:  # training/testing data available
        audio_paths = df["notated_path"].unique()
    else:
        # Data is in audio_directory
        audio_paths = ls_audio_dir(corpus)

    cepstra_paths = []
    for audio_path in audio_paths:
        audio_path = pathlib.Path(audio_path)

        if audio_path.suffix == ".npy":
            audio = np.load(str(audio_path))
            rate = corpus.config.sampling_rate
        else:
            audio, rate = lbr.load(audio_path, sr=config.sampling_rate)

        hop_length = seconds_to_audio(config.hop_length, rate)
        win_length = seconds_to_audio(config.win_length, rate)

        cepstrum = lbr.feature.mfcc(
            y=audio,
            sr=rate,
            n_mfcc=config.n_mfcc,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=config.n_fft,
            fmin=config.fmin,
            fmax=config.fmax,
            lifter=config.lifter,
        )

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

        # MFCCs are stored as structured arrays for convenience.
        # Arrays have fields:
        # notated_path: string - Audio file path
        # feature: float - corresponding MFCCs
        dtype = np.dtype(
            [
                ("notated_path", np.array(str(audio_path)).dtype),
                ("feature", cepstrum.dtype, cepstrum.shape),
            ]
        )
        data = np.zeros(1, dtype)
        data["notated_path"] = str(audio_path)
        data["feature"] = cepstrum

        notated_name = audio_path.stem
        mfcc_path = pathlib.Path(output_directory) / (notated_name + ".mfcc.npy")

        np.save(str(mfcc_path), data)

        cepstra_paths.append(
            {"notated_path": str(audio_path), "feature_path": str(mfcc_path)}
        )

    cepstrum_df = pd.DataFrame(cepstra_paths)

    corpus.register_data_resource(resource_name, cepstrum_df)

    return corpus
