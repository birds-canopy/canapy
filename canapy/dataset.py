import json
import math
import random
import glob
from pathlib import Path

import pandas as pd
import numpy as np

from joblib import Parallel, delayed
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import unique_labels

from .processor import Processor
from .config import default_config


def convert_memmap_dict(memmap_dict):
    def to_list(memmap_dict):
        if isinstance(memmap_dict, dict):
            return {k: to_list(v) for k, v in memmap_dict.items()}
        elif isinstance(memmap_dict, np.memmap):
            return memmap_dict.tolist()
        elif isinstance(memmap_dict, list):
            return memmap_dict

    return to_list(memmap_dict)


class Dataset(object):
    # Maximum authorized duration diff between samples to be considered as
    # consecutive, for merging purpose
    MAX_SILENCE_DIFF = -1

    @staticmethod
    def clean_index(df):
        if "start" in df.columns:
            df = df.sort_values(
                by=["wave", "start"], axis=0, ascending=True, ignore_index=True
            )
        else:
            df = df.sort_values(by=["wave"], axis=0, ascending=True, ignore_index=True)

        return df

    def __init__(
        self,
        directory,
        config=None,
        corrections=None,
        vocab=None,
        audioformat=None,
        rate=None,
        repertoire=None,
    ):
        self.df = None
        self.config = Config(**default_config())
        self.iteration = 0
        self.corrections = {self.iteration: {"syll": {}, "sample": {}}}
        self.vocab = None
        self.repertoire = None
        self.audioformat = audioformat

        print("format:", audioformat)

        self.directory = self._seek(directory)

        # Numpy arrays do not have sampling rate attached
        if self.audioformat == "npy" and rate is not None:
            self.config["sampling_rate"] = rate

        if config is not None:
            self.config = config
        if corrections is not None:
            self.corrections = corrections

        if self.df is not None:
            if self.df.get("syll") is not None:
                self.df = self.refine()
                self.df["syll"] = self.df["syll"].astype(str)
                self.vocab = unique_labels(self.df["syll"]).tolist()
            elif vocab:
                self.vocab = vocab
            elif self.vocab is None:
                raise ValueError(
                    "'vocab' can't be None if no annotations are available in the dataframe."
                )
        else:
            raise ValueError(
                f"{directory} is empty or no .csv file are in there. Can't build Dataset."
            )

        if self.config is None:
            raise ValueError(f"no 'config' file found in {directory}.")

        self.oh_encoder = OneHotEncoder(categories=[self.vocab], sparse=False)
        self.oh_encoder.fit(np.asarray(self.vocab).reshape(-1, 1))

        self.processor = Processor(self)

        self.annotations = dict()

    def _seek(self, directory):
        """
        Look for the data in directory
        """

        directory = Path(directory)

        if directory.exists():
            dfs = []
            audios = []
            audiofiles_suffix = "." + self.audioformat

            for f in directory.iterdir():
                if f.is_dir():
                    if "repertoire" in f.name:
                        self.repertoire = f
                    else:
                        for file in f.iterdir():
                            if file.suffix == ".csv":
                                dfs.append(pd.read_csv(file))
                            if file.suffix == audiofiles_suffix:
                                audios.append(file)
                                self.audiodir = f

                elif f.suffix == ".csv":
                    dfs.append(pd.read_csv(f))

                elif f.suffix == audiofiles_suffix:
                    audios.append(f)
                    self.audiodir = directory

                elif "correction" in f.name:
                    with f.open("r") as fileobj:
                        self.corrections = json.load(fileobj)
                    self.iteration = max(list(self.corrections.keys()))

                elif "config" in f.name:
                    with f.open("r") as fileobj:
                        self.config = Config(json.load(fileobj))

                elif "vocab" in f.name:
                    with f.open("r") as fileobj:
                        self.vocab = json.load(fileobj)

                if len(dfs) > 0:
                    self.df = pd.concat(dfs)
                else:
                    self.df = pd.DataFrame({"wave": [a.name for a in audios]})

                self.df = self.clean_index(self.df)
                self.original_df = self.df.copy(deep=True)
                self.audios = audios

            return directory

        else:
            raise NotADirectoryError(f"{directory} not found.")

    def _apply_corrections(self, df=None, corrections=None):
        """
        Apply corrections to DataFrame (class fusion and sample corrections).
        """
        corrections = self.corrections if corrections is None else corrections
        df = self.df if df is None else df

        if corrections is not None:
            # Merge classes
            if corrections.get("syll") is not None:
                df = df.replace(to_replace={"syll": corrections["syll"]})
            else:
                print("'syll' entry in corrections not found, skipping syll merge.")
            # Correct samples
            if corrections.get("sample") is not None:
                for index, corr in corrections["sample"].items():
                    df.at[int(index), "syll"] = corr
            else:
                print(
                    "'sample' entry in corrections not found, skipping sample correction."
                )

            # Delete samples
            df["syll"] = df["syll"].mask(df["syll"] == "DELETE", "SIL")

        df = self.clean_index(df)

        return df

    def _get_min_duration(self, config=None):
        """
        Compute the minimal duration authorized for an annotation, given
        the FFT window use for MFCC computation.
        """
        try:
            config = self.config if config is None else config
            win_length, min_analysis_win = config.get(
                "win_length", "min_analysis_window_per_sample"
            )
            return min_analysis_win * win_length
        except Exception:
            return 0.0

    def _remove_short_samples(self, df=None, config=None):
        """
        Remove samples that are too short to be analyzed with MFCC features.
        """

        df = self.df if df is None else df

        df["duration"] = df["end"] - df["start"]

        too_short = df["duration"] <= self._get_min_duration(config)
        if len(df[too_short]) > 0:
            print(
                f"{len(df[too_short])} annotations ignored - too short to be analysed \
                (< {self._get_min_duration()*1000}ms) : \
                    {list(df[too_short].groupby('syll').groups.keys())}"
            )

            df["syll"] = df["syll"].mask(too_short, "SIL")

        df.drop(["duration"], axis=1, inplace=True)

        df = self.clean_index(df)

        return df

    def _join_groups(self, df=None, config=None):
        """
        Join consecutive annotations if they are the same, and except for some that must not be joined.
        """

        df = self.df if df is None else df
        config = self.config if config is None else config

        excluded_sylls = config.keep_separate
        songs = df.groupby("wave").groups.keys()

        def _join(_df):
            _df = _df.reset_index(drop=True)
            _df["start_d"] = _df["start"].shift(-1)
            _df["diff"] = _df["start_d"] - _df["end"]
            first_consecutives = _df[
                (_df["syll"].shift(-1) == _df["syll"])
                & ~(_df["syll"].isin(excluded_sylls))
                & (_df["diff"] < Dataset.MAX_SILENCE_DIFF)
                & (_df["diff"] != np.nan)
            ]
            _df_c = _df.copy()
            for first in first_consecutives.itertuples():
                next = _df.index[first.Index + 1]
                _df_c.at[next, "start"] = first.start
                _df_c = _df_c.drop(first.Index, axis=0)

            return _df_c.drop(["start_d", "diff"], axis=1)

        with Parallel() as parallel:
            dfs = parallel(delayed(_join)(df[df["wave"] == s].copy()) for s in songs)

        df = pd.concat(dfs)
        df = self.clean_index(df)

        return df

    def _tag_silence(self, df=None):
        """
        Add a 'SIL' tag to every time gap between two samples in the dataset.
        """

        df = self.df if df is None else df

        # Loc silences
        df["s_start"] = df["end"].shift()
        df["s_end"] = df["start"]
        df = df.fillna(0)
        df.loc[df["s_end"] - df["s_start"] < -1, "s_start"] = 0.0
        df["duration"] = df["s_end"] - df["s_start"]

        # Remove irrelevant ones
        silence_samples = df.loc[df["duration"] > 0.0]
        silence_samples = silence_samples.drop(
            ["syll", "start", "end", "duration"], axis=1
        )
        silence_samples.reset_index(inplace=True, drop=True)

        # Tag silences
        silence_samples["syll"] = ["SIL"] * len(silence_samples)
        silence_samples = silence_samples[["wave", "s_start", "s_end", "syll"]]
        silence_samples.columns = ["wave", "start", "end", "syll"]

        # Clean up the mess
        df = df.drop(["s_start", "s_end", "duration"], axis=1)

        df = pd.concat([df, silence_samples]).reset_index(drop=True)

        return self.clean_index(df)

    def switch(self, directory):
        self.df = None
        self.audios = None
        self.audiodir = None
        self.directory = self._seek(directory)

        if self.df is not None:
            if self.df.get("syll") is not None:
                self.df = self.refine()
                self.df["syll"] = self.df["syll"].astype(str)
                self.vocab = unique_labels(self.df["syll"]).tolist()

    def update(self, iteration=None, corrections=None, checkpoint_to=None):
        if checkpoint_to is not None:
            self.checkpoint(checkpoint_to)

        if corrections is not None:
            self.iteration = iteration
            self.add_corrections(iteration, corrections)

        if self.df.get("syll") is not None:
            self.df = self.refine()
            self.df["syll"] = self.df["syll"].astype(str)
            self.vocab = unique_labels(self.df["syll"]).tolist()
            self.oh_encoder = OneHotEncoder(categories=[self.vocab], sparse=False)
            self.oh_encoder.fit(np.asarray(self.vocab).reshape(-1, 1))
        else:
            raise Exception("can't update a non annotated dataset.")

    def add_corrections(self, iteration, news, olds=None):
        corrections = self.corrections if olds is None else olds
        corrections[iteration] = news

    def refine(self, corrections=None, config=None):
        """
        Apply corrections to the dataset (class fusion, sample correction,
        very short samples removal, consecutive join)
        """
        corrections = self.corrections if corrections is None else corrections
        df = self.original_df.copy(deep=True)
        df = self._tag_silence(df)

        for correction in corrections.values():
            df = self._apply_corrections(df, corrections=correction)
            df = self._join_groups(df, config)
            df = self._remove_short_samples(df, config)

        return df

    def to_features(
        self,
        split=False,
        mode="syn",
        df=None,
        config=None,
        max_songs=None,
        return_dict=False,
    ):
        """
        Convert data to MFCC features and associated labels. If split is true,
        split the dataset between syn train, nsyn train and syn test and compute
        all features. Mode is ignored. If false, compute all features for the
        entire dataset, for syn or nsyn mode.
        """
        df = self.df if df is None else df
        config = self.config if config is None else config

        if split:
            train_songs, train_phrases, test_songs = self.to_trainset(
                df=df, config=config, max_songs=max_songs
            )

            test_feat = self.processor(test_songs, mode="syn", config=config)

            if mode == "syn" or mode is None:
                train_feat = self.processor(train_songs, mode="syn", config=config)
            elif mode == "nsyn":
                train_feat = self.processor(train_phrases, mode="nsyn", config=config)
            elif mode == "all":
                train_feat = self.processor(train_songs, mode="syn", config=config)
                ntrain_feat = self.processor(train_phrases, mode="nsyn", config=config)

                return train_feat, ntrain_feat, test_feat

            return train_feat, test_feat

        else:
            if max_songs:
                df = self.sample_songs(max_songs, df=df)
            return self.processor(df, mode=mode, config=config, return_dict=return_dict)

    def to_trainset(self, df=None, config=None, max_songs=None):
        """
        Returns split dataset for syn, non syn and ensemble training
        """
        train_songs, test_songs = self.split_syn(
            df=df, config=config, max_songs=max_songs
        )
        train_phrases = self.balance_nonsyn(df=train_songs, config=config)

        return train_songs, train_phrases, test_songs

    def sample_songs(self, max_songs, df=None, config=None):
        """Select a random set of 'max_songs' songs in the dataset"""
        df = df if df is not None else self.df
        config = self.config if config is None else config

        rs = np.random.RandomState(config.seed)
        songs = list(df.groupby("wave").groups.keys())
        selection = rs.choice(songs, size=max_songs, replace=False)
        reduced = df[df["wave"].isin(selection)]

        return reduced

    def balance_nonsyn(self, df=None, config=None):
        """Resample data to build a balanced in duration non syntactic set."""
        df = self.df if df is None else df
        config = self.config if config is None else config

        rs = np.random.RandomState(config.seed)

        min_duration, min_sil = config.get("min_class_duration", "min_silence_duration")

        # Select 'long' silences
        df["duration"] = df["end"] - df["start"]
        df = df[(df["syll"] != "SIL") | (df["duration"] > min_sil)]

        # Weight each sample in function of its class weight in the
        # dataset, in term of total duration, and regarding to the
        # min_duration objective
        durations = pd.DataFrame(df.groupby("syll")["duration"].sum())
        durations["weights"] = min_duration / durations["duration"]
        durations["weights"] = durations["weights"] / durations["weights"].sum()

        sub_rpz = durations[durations["duration"] < min_duration].index
        all_s_rpz = df[df["syll"].isin(sub_rpz)]

        all_weights = [durations.loc[s, "weights"] for s in all_s_rpz.syll]

        # resample the dataset with replace. Maximum probability weight is
        # given to under represented samples regarding to min_duration
        resampled = all_s_rpz
        N = len(all_s_rpz)
        while resampled.groupby("syll")["duration"].sum().min() < min_duration:
            resampled = all_s_rpz.sample(
                n=N, replace=True, weights=all_weights, random_state=rs
            )
            N += 5  # the balance is at a more or less ~5 samples precision
            # if too low, too slow...

        # mark duplicata as data to augment
        resampled["augmented"] = resampled.index.duplicated(keep="first")

        all_o_rpz = df[~df["syll"].isin(sub_rpz)]
        all_weights = [durations.loc[s, "weights"] for s in all_o_rpz.syll]

        # Same but with over represented samples (downsampling)
        subsampled = all_s_rpz
        N = len(all_s_rpz)
        while subsampled.groupby("syll")["duration"].sum().min() < min_duration:
            subsampled = all_o_rpz.sample(
                n=N, replace=True, weights=all_weights, random_state=rs
            )
            N += 5

        subsampled["augmented"] = subsampled.index.duplicated(keep="first")

        all_data = pd.concat([subsampled, resampled])
        all_data.drop(["duration"], axis=1, inplace=True)
        all_data.reset_index(drop=True, inplace=True)

        return all_data

    def split_syn(self, max_songs=None, df=None, config=None):
        """Build train and test sets from data for syntactic training.
        Ensure that at least one example of each syllable is present in train set.
        """
        df = self.df if df is None else df
        config = self.config if config is None else config

        rs = np.random.RandomState(config.seed)

        if max_songs and (max_songs < 0 or max_songs > len(df.groupby("wave"))):
            raise ValueError(f"can't select {max_songs} over {len(df.groupby('wave'))}")

        test_ratio = config.test_ratio

        # Select the least represented syllables
        minus = (
            df.groupby("syll")["syll"]
            .count()
            .index[df.groupby("syll")["syll"].count() <= 1]
        )
        minus_wave = df[df["syll"].isin(minus)].groupby("wave").count().index.values
        reduced_syn = df[df["wave"].isin(minus_wave)]

        # Put the songs containing them in train
        i = 1
        while len(reduced_syn.groupby("syll")) < len(self.vocab):
            i += 1
            minus = (
                df.groupby("syll")["syll"]
                .count()
                .index[df.groupby("syll")["syll"].count() < i]
            )
            minus_wave = df[df["syll"].isin(minus)].groupby("wave").count().index.values
            reduced_syn = df[df["wave"].isin(minus_wave)]

        print("Minimum number of songs necessary to train over all syllables :")
        print(len(reduced_syn.groupby("wave")))
        print("Total number of songs:")
        print(len(df.groupby("wave")))

        if (
            max_songs is not None
            and max_songs < len(reduced_syn.groupby("wave"))
            and max_songs > -1
        ):
            print(f"Warning : only {max_songs} will be selected.")

        already_picked = list(reduced_syn.groupby("wave")["wave"].groups.keys())
        left_to_pick = list(
            df[~df["wave"].isin(already_picked)].groupby("wave")["wave"].groups.keys()
        )

        more_size = math.floor(
            (1 - test_ratio) * len(left_to_pick) - test_ratio * len(already_picked)
        )

        some_more_songs = rs.choice(left_to_pick, size=more_size, replace=False)
        some_more_data = df[df["wave"].isin(some_more_songs)]

        reduced_syn = pd.concat([reduced_syn, some_more_data])

        already_picked = list(reduced_syn.groupby("wave")["wave"].groups.keys())
        left_to_pick = list(
            df[~df["wave"].isin(already_picked)].groupby("wave")["wave"].groups.keys()
        )
        test_syn = df[df["wave"].isin(left_to_pick)]

        if max_songs:
            reduced_syn = self.sample_songs(max_songs, df=reduced_syn, config=config)

        print("Final repartition of data :")
        print(
            f"Train : {len(reduced_syn.groupby('wave'))}, Test: {len(test_syn.groupby('wave'))}"
        )

        return reduced_syn, test_syn

    def checkpoint(self, directory):
        directory = Path(directory)
        if not (directory.exists()):
            directory.mkdir(parents=True)

        with Path(directory, "config.json").open("w+") as f:
            json.dump(self.config, f)
        if self.corrections is not None:
            with Path(directory, "corrections.json").open("w+") as f:
                json.dump(self.corrections, f)
        with Path(directory, "vocab.json").open("w+") as f:
            json.dump(self.vocab, f)

        if self.annotations is not None:
            with Path(directory, "annotations.json").open("w+") as f:
                json.dump(convert_memmap_dict(self.annotations), f)

    def export_config(self, file):
        with Path(file).open("w+") as f:
            json.dump(self.config, f)


class Config(dict):
    def __setattr__(self, attr, value):
        self[attr] = value

    def __getattr__(self, attr):
        try:
            if attr in ["syn", "nsyn"]:
                return Config(self[attr])
            if attr == "seed":
                if self["seed"] is None:
                    self["seed"] = random.randint(1, 9999)
            return self[attr]
        except KeyError:
            raise AttributeError(attr)

    def __repr__(self):
        return "Config " + super(Config, self).__repr__()

    def __str__(self):
        return "Config " + super(Config, self).__str__()

    def get(self, *items):
        if len(items) == 1:
            return super(Config, self).get(items[0])
        return tuple(super(Config, self).get(it) for it in items)

    def as_frames(self, attr):
        return round(self[attr] * self["sampling_rate"])

    def as_fftwindow(self, attr):
        if hasattr(self, "strict_window") and self["strict_window"] is True:
            return round(self[attr] * self["sampling_rate"])
        else:
            power = math.log(self[attr] * self["sampling_rate"]) / math.log(2)
            return 2 ** round(power)

    def as_duration(self, attr):
        return self[attr] / self["sampling_rate"]

    def duration(self, frames):
        """
        Compute the duration in seconds of a number of timesteps
        """
        return frames / self["sampling_rate"]

    def steps(self, duration):
        """
        Compute the number of discreet timestep contained in a time in seconds
        """
        return round(duration * self["sampling_rate"])

    def frames(self, duration):
        """
        Compute the number of MFCC windows contained in a time in seconds.
        """
        return round(
            round(duration * self["sampling_rate"]) / self.as_fftwindow("hop_length")
        )

    def to_duration(self, mfcc_frames):
        return mfcc_frames * self.as_fftwindow("hop_length") / self["sampling_rate"]


def update_corrections(self, news, olds=None):
    corrections = self.corrections if olds is None else olds
    if corrections is not None:
        old_syll = corrections["syll"].copy()
        old_sample = corrections["sample"].copy()

        old_syll.update(news["syll"])
        old_sample.update(news["sample"])

        still = True
        while still:
            still = False
            for k, v in old_syll.items():
                if old_syll.get(v) and not old_syll.get(v) == v:
                    old_syll[k] = old_syll[v]
                    still = True

        updated = {
            "syll": {k: v for k, v in old_syll.items() if k != v and v is not None}
        }

        updated["sample"] = {
            k: updated["syll"][v] if updated["syll"].get(v) is not None else v
            for k, v in old_sample.items()
        }

        return updated
