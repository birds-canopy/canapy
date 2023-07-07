# Author: Nathan Trouvain at 27/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
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
        return len(self.dataset["annotation"].unique())

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
    def from_df(cls, df, annots_directory=None, config=None, seq_ids=None):
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
