# Author: Nathan Trouvain at 28/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import logging
from pathlib import Path

import numpy as np
import librosa as lbr
import pandas as pd

from ...timings import seconds_to_audio
from ...log import log


logger = logging.getLogger("canapy")


class AudioNotFound(Exception):
    pass


def ls_audio_dir(audio_directory, audio_ext):
    audio_dir = Path(audio_directory)
    return list(audio_dir.rglob(f"**/*{audio_ext}"))


def ls_spec_dir(spec_directory, spec_ext):
    spec_dir = Path(spec_directory)

    if not spec_dir.exists():
        return pd.DataFrame(columns=["notated_path", "feature_path"])

    spec_paths = list(spec_dir.rglob(f"**/*{spec_ext}"))

    if len(spec_paths) == 0:
        return pd.DataFrame(columns=["notated_path", "feature_path"])

    no_audio_file = []
    spec_registry = []
    for spec_path in spec_paths:
        spec = np.load(str(spec_path))

        if hasattr(spec, "keys") and "notated_path" in spec:
            notated_path = str(spec["notated_path"])
            spec_registry.append(
                {"notated_path": notated_path, "feature_path": spec_path}
            )
        else:
            no_audio_file.append(spec_path)
            spec_registry.append({"notated_path": np.nan, "feature_path": spec_path})

    if len(no_audio_file) > 0:
        logger.warning(
            f"Found {len(no_audio_file)} spectrograms or feature files with no "
            f"corresponding audio file. If this is not the expected "
            f"behavior and you are providing spectrograms or features as "
            f"Numpy archive arrays with two fields: 'notated_path' "
            f"storing the name of the corresponding audio file "
            "and 'feature' storing the spectrogram data."
        )

    return pd.DataFrame(spec_registry)


def load_audio(audio_path, sampling_rate):
    audio_path = Path(audio_path)

    if audio_path.suffix == ".npy":
        audio = np.load(str(audio_path))
        if audio.ndim == 2:  # more than one channel
            if (
                audio.shape[1] < 3 and audio.shape[0] > 3
            ):  # time should always be second dimension
                audio = audio.T
            audio = np.mean(audio, axis=0).flatten()  # converting to mono
        elif audio.ndim > 2:
            raise ValueError(
                f"Audio {audio_path} dimension is {audio.shape}. "
                "Audio array should be one or two dimensional."
            )
    else:
        audio, _ = lbr.load(audio_path, sr=sampling_rate)

    return audio


def get_filenames(paths):
    # Additional split in case file name as two exts
    return [Path(p).stem.split(".")[0] for p in paths]


@log(fn_type="audio transform")
def compute_mfcc(corpus, *, output_directory, resource_name, redo=False, **kwargs):

    if redo and corpus.audio_directory is None:
        raise AudioNotFound(
            "Can't redo spectrograms if no audio is provided! (audio_directory is None)"
        )

    df = corpus.dataset
    config = corpus.config.transforms.audio

    spec_path = (
        corpus.spec_directory if output_directory is None else Path(output_directory)
    )

    cepstrum_df = ls_spec_dir(spec_path, spec_ext=corpus.spec_ext)

    if len(df) > 0:  # training/testing data available
        audio_paths = df["notated_path"].unique()
    else:
        # Data is in audio_directory
        audio_paths = ls_audio_dir(corpus.audio_directory, corpus.audio_ext)

    # Try loading spectrograms
    if len(cepstrum_df) > 0 and not redo:
        cep_names = set(get_filenames(cepstrum_df["notated_path"].unique()))

        # If we already have the resources registered in the corpus,
        # check the corpus has correct references
        if resource_name in corpus.data_resources:
            curr_ressources = corpus.data_resources[resource_name]
            notated_names = curr_ressources["notated_path"].unique()
            if set(get_filenames(notated_names)) <= cep_names:
                logger.info("Found previously computed spectrograms. Will use them.")
                return corpus
            else:
                logger.warn("Not all spectrograms could be found. Recomputing.")

        # If we have audios, check we have corresponding spectrograms
        if len(audio_paths) > 0:
            audio_names = set(get_filenames(audio_paths))
            if audio_names <= cep_names:
                resource = cepstrum_df.query("notated_path in @audio_paths")
                corpus.register_data_resource(resource_name, resource)
                logger.info(
                    "Found matching spectrograms and audio. Will use spectrograms."
                )
                return corpus
            else:
                logger.warn("Spectrograms do not match audio files. Recomputing.")

        # If we only have MFCCs:
        else:
            corpus.register_data_resource(resource_name, cepstrum_df)
            logger.info("Using spectrograms. No audio data.")
            return corpus

    elif len(audio_paths) == 0:
        audio_ext = corpus.audio_ext
        audio_dir = corpus.audio_directory

        raise AudioNotFound(
            f"No spectrograms provided, and no audio data file with extension '{audio_ext}' "
            f"found in {audio_dir}. Consider changing the 'audio_ext' parameter when loading your Corpus, "
            f"if your audio data is not in '{audio_ext} format, or checking that {audio_dir} "
            f"is not empty."
        )

    logger.info("Computing spectrograms...")

    cepstra_paths = []
    for audio_path in audio_paths:

        rate = config.sampling_rate
        audio = load_audio(audio_path, rate)

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

        data = {
            "notated_path": str(audio_path),
            "feature": cepstrum,
        }

        notated_name = Path(audio_path).stem
        mfcc_path = spec_path / (notated_name + corpus.spec_ext)

        np.savez(str(mfcc_path), **data)

        cepstra_paths.append(
            {"notated_path": str(audio_path), "feature_path": str(mfcc_path)}
        )

    cepstrum_df = pd.DataFrame(cepstra_paths)

    corpus.register_data_resource(resource_name, cepstrum_df)

    return corpus
