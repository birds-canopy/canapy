# Author: Nathan Trouvain at 27/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
"""Example NumPy style docstrings.

This module provides the Corpus class which store canary songs data

Example
-------

    >>> my_corpus = Corpus.from_directory(
    >>>     audio_directory="home/vincent/Documents/data_canary/audio",
    >>>     spec_directory="home/vincent/Documents/data_canary/spec",
    >>>     annots_directory="home/vincent/Documents/data_canary/annotations",
    >>>     config_path="home/vincent/Documents/data_canary/config",
    >>> )
    >>> print(my_corpus.dataset["label"])


Section breaks are created with two blank lines. Section breaks are also
implicitly created anytime a new section starts. Section bodies *may* be
indented:

Notes
-----
    This is an example of an indented section. It's like any other section,
    but the body is indented to help it stand out from surrounding text.

If a section is indented, then a section break is created by
resuming unindented text.

Attributes
----------


"""
import pathlib
from pathlib import Path
from typing import Iterable, Optional, Union, Sequence, Dict, Any

import attr
import pandas as pd
import numpy as np

import crowsetta
from crowsetta.formats.seq import GenericSeq

from .config import Config, default_config
from .utils import as_path


@attr.define
class Corpus:
    audio_directory: Optional[Union[Path, str]] = attr.field(converter=as_path)
    spec_directory: Optional[Union[Path, str]] = attr.field(converter=as_path)
    annots_directory: Optional[Union[Path, str]] = attr.field(converter=as_path)
    annot_format: str = attr.field(default="marron1csv")
    annotations: GenericSeq = attr.field(default=GenericSeq(annots=list()))
    config: Config = attr.field(default=default_config)
    dataset: Optional[pd.DataFrame] = attr.field(default=None)
    data_resources: Optional[Dict[str, Any]] = attr.field(default=dict())
    audio_ext: Optional[str] = attr.field(default=".wav")
    spec_ext: Optional[str] = attr.field(default=".npy")

    def __attrs_post_init__(self):
        if isinstance(self.annotations.annots, Sequence):
            n_annots = len(self.annotations.annots)
        else:
            n_annots = 1

        if n_annots == 0:
            self.dataset = pd.DataFrame(
                columns=[
                    "label",
                    "onset_s",
                    "offset_s",
                    "notated_path",
                    "annot_path",
                    "sequence",
                    "annotation",
                ]
            )
        else:
            self.dataset = self.annotations.to_df().sort_values(
                by=["annotation", "sequence", "onset_s"]
            )

        self.data_resources = dict()

    def __len__(self):
        """
        Give the length of the dataset stored

        Returns
        -------
        int
            length of the DataFrame
        """
        return len(self.dataset["annotation"].unique())

    def __getitem__(self, item):
        return self.clone_with_df(self.dataset.query(item))


    @classmethod
    def from_directory(
        cls,
        audio_directory=None,
        spec_directory=None,
        annots_directory=None,
        config_path=None,
        annot_format="marron1csv",
        time_precision=0.001,
        audio_ext=".wav",
        spec_ext=".mfcc.npy",
    ):
        """
        Create a Corpus object from audios, annotations, and spec stored on the disk.

        Parameters
        ----------
        audio_directory : str , optional
            path of the directory which contains the audio tracks
        spec_directory : str , optional
            path of the directory which contains the spectra files
        annots_directory : str , optional
            path of the directory which contains hand-made annotations
        config_path : str , optional
            path of the directory which contains the configuration
            by default, default_config (from config.py or config.toml) will be applied
        annot_format : str , default : "marron1csv
            <complete>
        time_precision : float , default : 0.001
            <complete>
        audio_ext : str or tuple of str , default : ".wav"
            extension for the audio files in the audio_directory
        spec_ext : str , default : ".mfcc.npy"
            extension for the spec files in the spec_directory

        Returns
        -------
        Corpus
            Contains the audio, annotations and spec files from the given directories

        Raises
        ------
        ValueError
            At least one of audio_directory or spec_directory must be provided.
        """
        if audio_directory is None and spec_directory is None:
            raise ValueError(
                "At least one of audio_directory or spec_directory must " "be provided."
            )

        audio_dir = as_path(audio_directory)
        spec_dir = as_path(spec_directory)
        annots_dir = None

        annotations = GenericSeq(annots=list())

        if annots_directory is not None:
            annots_dir = Path(annots_directory)

            scribe = crowsetta.Transcriber(format=annot_format)
            annot_ext = crowsetta.formats.by_name(annot_format).ext

            # annot_ext might be a tuple of admissible file extensions
            if isinstance(annot_ext, str):
                annot_files = sorted(annots_dir.rglob(f"**/*{annot_ext}"))
            else:
                annot_files = list()
                for ext in annot_ext:
                    annot_files += list(annots_dir.rglob(f"**/*{ext}"))
                annot_files = sorted(annot_files)

            annotations = list()
            for annot_file in annot_files:
                annots = scribe.from_file(
                    annot_path=annot_file,
                    notated_path=audio_dir,
                ).to_annot(decimals=round(-np.log10(time_precision)))

                if isinstance(annots, Iterable):
                    annotations.extend(annots)
                else:
                    annotations.append(annots)

            annotations = GenericSeq(annots=annotations)

        if config_path is not None:
            config = Config.from_file(config_path)
        else:
            config = default_config

        return cls(
            audio_directory=audio_dir,
            spec_directory=spec_dir,
            annots_directory=annots_dir,
            annot_format=annot_format,
            annotations=annotations,
            config=config,
            audio_ext=audio_ext,
            spec_ext=spec_ext,
        )

    @classmethod
    def from_df(cls, df, annots_directory=None, config=default_config, seq_ids=None):
        """
        Create a Corpus object from a DataFrame which contains data about canary songs.

        Parameters
        ----------
        df : pandas.DataFrame
            contains data about one or more canary songs
        annots_directory : str , optional
            path of the directory which contains hand-made annotations
        config : config.Config , default : default_config (from config.py or config.toml)
            contains parameters of the corpus
        seq_ids : , optional
            <complete>

        Returns
        -------
        Corpus
            Contains data from 'df'

        Raises
        ------
        ValueError
            'seq_ids' should have same length than 'df'.
        """
        seq_ids = np.sort(df["notated_path"]) if seq_ids is None else seq_ids

        if len(seq_ids) != len(df):
            raise ValueError("'seq_ids' should have same length than 'df'.")

        annots_dir = as_path(annots_directory)

        annotation_list = list()
        for seq_id in np.unique(seq_ids):
            seq_idx = np.where(seq_ids == seq_id, True, False)
            annots = df[seq_idx].sort_values(by="onset_s")

            seq = crowsetta.Sequence.from_keyword(
                labels=annots.label, onsets_s=annots.onset_s, offsets_s=annots.offset_s
            )

            notated_path = annots.notated_path.unique()
            if len(notated_path) > 1:
                raise ValueError(
                    f"Some annotations are affected to more "
                    f"than one notated file: {annots}"
                )

            notated_path = pathlib.Path(notated_path[0])

            if "annot_path" in annots:
                annot_path = annots["annot_path"].unique()[0]
            elif annots_directory is not None:
                annot_path = annots_dir / (notated_path.stem + ".csv")
            else:
                annot_path = notated_path.with_suffix(".csv")

            annot = crowsetta.Annotation(
                annot_path=annot_path, notated_path=notated_path, seq=seq
            )

            annotation_list.append(annot)

        annotations = GenericSeq(annots=annotation_list)

        return cls(
            audio_directory=None,
            spec_directory=None,
            annots_directory=annots_dir,
            annot_format="generic-seq",
            annotations=annotations,
            config=config,
            audio_ext=None,
            spec_ext=None,
        )

    def to_directory(self, annots_directory):
        """
        Store the annotations on the disk
        Annotations will be stored on csv format and in the following form :
            wave  -> name of the audio track it cames from
            start -> time marker of the begining of the phrase (in seconds)
            end   -> time marker of the end of the phrase (in seconds)
            syll  -> class of the phrase

        Parameters
        ----------
        annots_directory : str
            path where the annotations will be stored

        Returns
        -------
        Corpus
            same corpus

        """
        Path(annots_directory).mkdir(parents=True, exist_ok=True)

        if not isinstance(self.annotations.annots, Sequence):
            annotations = [self.annotations.annots]
        else:
            annotations = self.annotations.annots

        for annots in annotations:
            seq = GenericSeq(annots=annots)
            annot_path = annots.annot_path
            annot_path = Path(annots_directory) / pathlib.Path(annot_path).name

            annots_df = seq.to_df(basename=True)

            # crowsetta makes it difficult to save to various formats.
            # We will keep using marron1csv format by default for now.

            annots_df = pd.DataFrame(
                {
                    "wave": annots_df.notated_path,
                    "start": annots_df.onset_s,
                    "end": annots_df.offset_s,
                    "syll": annots_df.label,
                }
            )

            annots_df.to_csv(annot_path, index=False)

        return self

    def register_data_resource(self, name, data):
        """
        Add a new data ressource to a corpus

        Parameters
        ----------
        name : str
            name of the new data ressource
        data : any
            data of the new data ressource

        Returns
        -------
        Corpus
            Same corpus with the new data ressource added
        """
        self.data_resources[name] = data
        return self

    def clone_with_df(self, df):
        new_corpus = Corpus.from_df(
            df, annots_directory=self.annots_directory, config=self.config
        )

        new_corpus.audio_directory = self.audio_directory
        new_corpus.spec_directory = self.spec_directory
        new_corpus.annot_format = self.annot_format
        new_corpus.audio_ext = self.audio_ext
        new_corpus.spec_ext = self.spec_ext

        # Do not copy data_resources! All clones should be able to access
        # heavy transformed data without redoing the transform.
        new_corpus.data_resources = self.data_resources

        return new_corpus
