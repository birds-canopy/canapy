# Authors: Nathan Trouvain at 27/06/2023 <nathan.trouvain<at>inria.fr>
#          Vincent Gardies at 12/07/2023 <vincent.gardies<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain

"""
Provides the Corpus class for storing canary songs' data.

A corpus can be created from a repository containing audio, annotation, and spectrogram files,
or from a DataFrame that already contains data about canary songs.
Annotations from a corpus can be saved to disk using the 'to_directory' method.

Example
-------
    >>> from canapy.corpus import Corpus
    >>> # Import the Corpus class
    >>> my_corpus_audio_and_annot = Corpus.from_directory(
    >>>     audio_directory="/home/vincent/Documents/data_canary/audio",
    >>>     annots_directory="/home/vincent/Documents/data_canary/annotations"
    >>> )
    >>> # Create a new corpus using the audio and annotations files from the 'data_canary' folder
    >>> my_corpus_audio = Corpus.from_directory(audio_directory="home/vincent/Documents/data_canary/audio")
    >>> # Create a new corpus using only audio files
    >>> print(my_corpus_audio_and_annot.dataset)
    >>> # Print the data of 'my_corpus_audio_and_annot'
    >>> print(len(my_corpus_audio_and_annot))
    >>> # Print the number of songs in the corpus
    >>> print(my_corpus_audio_and_annot["label != 'cri'"].dataset)
    >>> # Print the data of a new corpus where there are no 'cri' annotations

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
    """
    Store canary songs' data.
    Standard form to perform annotators prediction and training over canary songs' data.

    Attributes
    ----------
    audio_directory : pathlib.Path | str
        Path of the directory that contains audio files.
    spec_directory : pathlib.Path | str
        Path of the directory that contains spectrogram files.
    annots_directory : pathlib.Path | str
        Path of the directory that contains annotation files.
    annot_format : str
        Format of the annotation data. <verify>
    annotations : GenericSeq (from crowsetta)
        Represents annotations from a generic format, meant to be an abstraction of any sequence-like format.
    config : Config (from config.toml)
        Store every parameter about corpus, transformations and annotators.
    dataset : pd.DataFrame
        The DataFrame that stores annotation data.
    data_resources : dict
        Additional resources about the corpus, such as applied transformations.
    audio_ext : str
        Extension of audio files in audio_directory.
    spec_ext : str
        Extension of spectrogram files in spec_directory.

    Methods
    -------
    from_directory(audio_directory=None, spec_directory=None, annots_directory=None, config_path=None,
                   annot_format="marron1csv", time_precision=0.001, audio_ext=".wav", spec_ext=".mfcc.npy")
        Create a Corpus object from audios, annotations, and spectrogram files stored on the disk.
    from_df(df, annots_directory=None, config=default_config, seq_ids=None)
        Create a Corpus object from a DataFrame that contains data about canary songs.
    to_directory(self, annots_directory)
        Store the annotations on the disk.

    """

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
        """
        Post-initialization method.

        If the annotations are provided as a sequence, convert them to a DataFrame and sort them.
        Otherwise, create an empty DataFrame with the required columns.
        """
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
        Give the number of songs in the Corpus.

        Returns
        -------
        int
            Number of songs in the Corpus.

        Example
        -------
            >>> corpus = Corpus(...)
            >>> len(corpus) # number of songs in the corpus
        """
        return len(self.dataset["annotation"].unique())

    def __getitem__(self, item):
        """
        Create a new corpus whose dataset satisfies the query.

        Parameters
        ----------
        item : str
            The query string to evaluate. Refer to the pandas.DataFrame.query documentation for details.

        Returns
        -------
        Corpus
            New corpus which dataset contains the query applied to the former dataset.

        Examples
        --------
            >>> corpus = Corpus (...)
            >>> # 'corpus' is the original corpus
            >>> corpus_without_cri = corpus["label != 'cri'"]
            >>> # corpus_without_cri is a copy of corpus where every line of the dataset with the label 'cri' is erased

            >>> corpus_first_seconds = corpus["offset_s <= 10"]
            >>> # corpus_first_seconds is a copy of corpus where there is only line that offset_s is smaller than 10
            >>> # so it contains the annotation that stop before the first 10 seconds

            >>> corpus_long_phrase = corpus["offset_s - onset_s > 1"]
            >>> # corpus_long_phrase is a copy of corpus where every phrase of the dataset that last less than a second
            >>> # are erased
        """
        print(item)
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
        Create a Corpus object from audios, annotations, and spectrogram files stored on the disk.

        Parameters
        ----------
        audio_directory : str, optional
            Path of the directory that contains the audio tracks.
        spec_directory : str, optional
            Path of the directory that contains the spectrogram files.
        annots_directory : str, optional
            Path of the directory that contains hand-made annotations.
        config_path : str, optional
            Path of the directory that contains the configuration.
            By default, default_config (from config.py or config.toml) will be applied.
        annot_format : str, default="marron1csv"
            The format of the annotation data.
        time_precision : float, default=0.001
            The time precision.
        audio_ext : str, default=".wav"
            The extension for the audio files in the audio_directory.
        spec_ext : str, default=".mfcc.npy"
            The extension for the spectrogram files in the spec_directory.

        Returns
        -------
        Corpus
            Corpus object that contains the audio, annotations, and spectrogram files from the given directories.
        Raises
        ------
        ValueError
            At least one of audio_directory or spec_directory must be provided.

        Notes
        -----
        Repertories do not have to be distinct

        Examples
        --------
            >>> corpus_audio = Corpus(audio_directory="/home/vincent/Documents/data_canary/audio")
            >>> # corpus_audio is a corpus with only audio tracks

            >>> corpus_audio_annotation = Corpus(
            >>>     audio_directory="/home/vincent/Documents/data_canary/mix_audio_annots",
            >>>     annots_directory="/home/vincent/Documents/data_canary/mix_audio_annots"
            >>> ) # This corpus is made with the audio and annotation files in the 'mix_audio_annots' folder

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
        Create a Corpus object from a DataFrame that contains data about canary songs.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that contains data about one or more canary songs.
        annots_directory : str, optional
            Path of the directory that contains hand-made annotations.
        config : Config, default=default_config (from canapy.config)
            Contains parameters of the corpus.
        seq_ids : Iterable, optional
            Sequence identifiers.

        Returns
        -------
        Corpus
            Corpus object that contains data from 'df'.

        Raises
        ------
        ValueError
            If 'seq_ids' should have same length than 'df'.

            If some annotations are affected to more than one notated file.
        """
        seq_ids = df["notated_path"] if seq_ids is None else seq_ids

        if len(seq_ids) != len(df):
            raise ValueError("'seq_ids' should have same length than 'df'.")

        annots_dir = as_path(annots_directory)

        annotation_list = list()
        for seq_id, annots in df.groupby(seq_ids):
            # seq_idx = np.where(seq_ids == seq_id, True, False)
            # annots = df[seq_idx].sort_values(by="onset_s")

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
        Store the annotations on the disk.

        Annotations will be stored in CSV format with the following columns:
            - 'wave' : Name of the audio track it came from.
            - 'start': Time marker of the beginning of the phrase (in seconds).
            - 'end'  : Time marker of the end of the phrase (in seconds).
            - 'syll' : Class of the phrase.

        Parameters
        ----------
        annots_directory : str
            Path where the annotations will be stored.

        Returns
        -------
        Corpus
            Same corpus object

        Example
        -------
            >>> corpus_canapy_predictions.to_directory("/home/vincent/Documents/data_canary/canapy_predictions")
            >>> # predictions made by canapy are now stored in the disk in the 'canapy_predictions' folder
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
        Add a new data ressource to the corpus.

        Parameters
        ----------
        name : str
            Name of the new data resource.
        data : Any
            Data of the new data resource.

        Returns
        -------
        Corpus
            Same corpus object with the new data resource added.
        """
        self.data_resources[name] = data
        return self

    def clone_with_df(self, df):
        """<verify>
        Create a copy of the corpus with a new dataset

        Parameters
        ----------
        df : pd.DataFrame
            Dataset for the new corpus.

        Returns
        -------
        Corpus
            Copy of the original corpus with a new dataset, 'df'.

        """

        if "annotation" in df and "sequence" in df:
            seq_ids = df["annotation"].astype(str) + df["sequence"].astype(str)
        else:
            seq_ids = None

        new_corpus = Corpus.from_df(
            df, annots_directory=self.annots_directory, config=self.config, seq_ids=seq_ids,
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
