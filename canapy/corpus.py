# Author: Nathan Trouvain at 27/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
from pathlib import Path
from typing import Iterable, Optional, Union, Sequence, Dict, Any

import attr
import pandas as pd
import numpy as np

import crowsetta
from crowsetta.formats.seq import GenericSeq

from .config import Config, default_config
from .formats.marron1csv import Marron1CSV

from .transforms.commons.training import encode_labels


def genericseq_from_df(df):
    ...


def as_path(path_or_none):
    if path_or_none is not None:
        return Path(path_or_none)
    else:
        return path_or_none


@attr.define
class Corpus:
    audio_directory: Union[Path, str] = attr.field(converter=as_path)
    spec_directory: Optional[Union[Path, str]] = attr.field(converter=as_path)
    annots_directory: Optional[Union[Path, str]] = attr.field(converter=as_path)
    annot_format: str = attr.field(default="marron1csv")
    annotations: GenericSeq = attr.field(default=GenericSeq(annots=list()))
    config: Config = attr.field(default=default_config)
    dataset: Optional[pd.DataFrame] = attr.field(default=None)
    data_resources: Optional[Dict[str, Any]] = attr.field(default=dict())

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
            self.dataset = self.annotations.to_df()

        self.data_resources = dict()

    def __len__(self):
        return len(self.dataset["annotation"].unique())

    @classmethod
    def from_directory(
        cls,
        audio_directory,
        spec_directory=None,
        annots_directory=None,
        config_path=None,
        annot_format="marron1csv",
        time_precision=0.001,
    ):
        audio_dir = Path(audio_directory)
        annots_dir = None
        annotations = GenericSeq(annots=list())

        if annots_directory is not None:
            annots_dir = Path(annots_directory)

            scribe = crowsetta.Transcriber(format=annot_format)
            annot_ext = crowsetta.formats.by_name(annot_format).ext

            annotations = list()
            for annot_file in annots_dir.rglob(f"**/*{annot_ext}"):
                annots = scribe.from_file(
                    annot_path=annot_file, notated_path=audio_dir
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

        spec_dir = None
        if spec_directory is not None:
            spec_dir = Path(spec_directory)

        return cls(
            audio_directory=audio_dir,
            spec_directory=spec_dir,
            annots_directory=annots_dir,
            annot_format=annot_format,
            annotations=annotations,
            config=config,
        )

    def register_data_resource(self, name, data):
        self.data_resources[name] = data

    def save(self, directory, annot_format="marron1csv"):
        genericseq_from_df(self.dataset)


if __name__ == "__main__":
    crowsetta.register_format(Marron1CSV)
    c = Corpus.from_directory(
        audio_directory="/home/nathan/Documents/Code/canapy-test/data/",
        annots_directory="/home/nathan/Documents/Code/canapy-test/data/",
    )

    df = c.dataset

    c = encode_labels(c)

    print(c)
