# Author: Nathan Trouvain at 27/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import pathlib

import numpy as np
import pandas as pd
import librosa as lbr

from .base import Transform
from .commons.audio import compute_mfcc
from .commons.training import prepare_dataset_for_training, encode_labels
from ..log import log
from ..timings import seconds_to_audio


def _noisify(audio, config, rs):
    """
    Add random perturbation to an audio signal.
    """
    mu = audio.mean()
    sigma = audio.std()

    audio = (audio - mu) / sigma

    # additive noise
    add_noise = rs.normal(
        0.0,
        config.transforms.training.balance.data_augmentation.noise_std,
        size=audio.shape,
    )
    audio = audio + add_noise

    audio = audio * sigma + mu

    return audio


@log(fn_type="training data tranform")
def balance_labels_duration(corpus, *, resource_name, **kwargs):
    """Resample data to build a balanced in duration non-syntactic set."""

    # If the dataset is already split, then only balance the training set
    df = corpus.dataset.query("train")

    config = corpus.config.transforms.training.balance

    rs = np.random.RandomState(corpus.config.misc.seed)

    min_duration = config.min_class_total_duration
    min_silence = config.min_silence_duration
    silence_tag = corpus.config.transforms.annots.silence_tag

    balanced_df = df.copy(deep=True)

    # Select 'long' silences
    balanced_df["duration"] = balanced_df["offset_s"] - balanced_df["onset_s"]

    balanced_df = balanced_df.query("label != @silence_tag or duration > @min_silence")

    # Weight each sample in function of its class weight in the
    # dataset, in terms of total duration, and regarding the
    # min_duration objective
    durations = pd.DataFrame(balanced_df.groupby("label")["duration"].sum())
    durations["weights"] = min_duration / durations["duration"]
    durations["weights"] = durations["weights"] / durations["weights"].sum()

    underrepresented_cls = durations.query("duration < @min_duration").index
    underrepresented = balanced_df.query("label in @underrepresented_cls")

    weights = [durations.loc[s, "weights"] for s in underrepresented.label]

    # resample the dataset with replace. Maximum probability weight is
    # given to underrepresented samples regarding min_duration
    resampled = underrepresented
    N = len(underrepresented)
    while resampled.groupby("label")["duration"].sum().min() < min_duration:
        resampled = underrepresented.sample(
            n=N, replace=True, weights=weights, random_state=rs
        )
        N += 5  # the balance is at a more or less ~5 samples precision
        # if too low, too slow...

    # mark duplicata as data to augment
    resampled["augmented"] = resampled.index.duplicated(keep="first")

    overrepresented = balanced_df.query("label not in @underrepresented_cls")
    weights = [durations.loc[s, "weights"] for s in overrepresented.label]

    # Same but with over-represented samples (downsampling)
    subsampled = overrepresented
    N = len(overrepresented)
    while subsampled.groupby("label")["duration"].sum().min() < min_duration:
        subsampled = overrepresented.sample(
            n=N, replace=True, weights=weights, random_state=rs
        )
        N += 5

    subsampled["augmented"] = subsampled.index.duplicated(keep="first")

    balanced_df = pd.concat([subsampled, resampled])
    balanced_df.drop(["duration"], axis=1, inplace=True)
    balanced_df.reset_index(drop=True, inplace=True)

    corpus.register_data_resource(resource_name, balanced_df)

    return corpus


@log(fn_type="training data transform")
def compute_mfcc_for_balanced_dataset(corpus, *, resource_name, redo=False, **kwargs):
    df = corpus.data_resources.get("balanced_dataset", corpus.dataset)
    config = corpus.config.transforms.audio

    rs = np.random.default_rng(corpus.config.misc.seed)

    # if resource_name in corpus.data_resources and not redo:
    #     return corpus

    audio_paths = df["notated_path"].unique()

    mfcc = [np.nan] * df.shape[0]
    for audio_path_ in audio_paths:
        audio_path = pathlib.Path(audio_path_)

        if audio_path.suffix == ".npy":
            audio = np.load(str(audio_path))
            rate = corpus.config.sampling_rate
        else:
            audio, rate = lbr.load(audio_path, sr=config.sampling_rate)

        annots = df.query("notated_path == @audio_path_")

        hop_length = seconds_to_audio(config.hop_length, rate)
        win_length = seconds_to_audio(config.win_length, rate)

        for entry in annots.itertuples():
            start = seconds_to_audio(entry.onset_s, rate)
            end = seconds_to_audio(entry.offset_s, rate)
            one_label = audio[start:end]
            if hasattr(entry, "augmented"):
                one_label = _noisify(one_label, corpus.config, rs)

            cepstrum = lbr.feature.mfcc(
                y=one_label,
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

            mfcc[entry.Index] = cepstrum

    df["mfcc"] = mfcc

    corpus.register_data_resource(resource_name, df)

    return corpus


class NSynESNTransform(Transform):
    def __init__(self):
        super().__init__(
            training_data_transforms=[
                prepare_dataset_for_training,
                balance_labels_duration,
                compute_mfcc_for_balanced_dataset,
            ],
            training_data_resource_name=[
                "dataset",
                "balanced_dataset",
                "mfcc_dataset",
            ],
            audio_transforms=[compute_mfcc],
            audio_resource_names=["syn_mfcc"],
        )
