# Author: Nathan Trouvain at 27/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import abc
import pathlib

from typing import Callable, Mapping, Iterable, Optional

import attr


@attr.define
class Transform(abc.ABC):
    annots_transforms: Optional[Iterable[Callable]] = attr.field(default=list())
    training_data_transforms: Optional[Iterable[Callable]] = attr.field(default=list())
    training_data_resource_name: Optional[Iterable[str]] = attr.field(default=list())
    audio_transforms: Optional[Iterable[Callable]] = attr.field(default=list())
    audio_resource_names: Optional[Iterable[str]] = attr.field(default=list())

    def __call__(
        self,
        corpus,
        purpose="training",
        redo_annots=False,
        redo_audio=False,
        redo_training=False,
        output_directory=None,
    ):
        if purpose == "training":
            corpus = self.transform_annots(corpus, redo=redo_annots)
            corpus = self.transform_audio(
                corpus, redo=redo_audio, output_directory=output_directory
            )
            corpus = self.transform_training_data(corpus, redo=redo_training)

        elif purpose == "annotation":
            corpus = self.transform_annots(corpus, redo=redo_annots)
            corpus = self.transform_audio(
                corpus, redo=redo_audio, output_directory=output_directory
            )

        return corpus

    def transform_annots(self, corpus, redo=False):
        for transform in self.annots_transforms:
            corpus = transform(corpus, redo=redo)
        return corpus

    def transform_training_data(self, corpus, redo=False):
        for transform, resource_name in zip(
            self.training_data_transforms, self.training_data_resource_name
        ):
            corpus = transform(corpus, resource_name=resource_name, redo=redo)
        return corpus

    def transform_audio(self, corpus, redo=False, output_directory=None):
        for transform, resource_name in zip(
            self.audio_transforms, self.audio_resource_names
        ):
            corpus = transform(
                corpus,
                output_directory=output_directory,
                resource_name=resource_name,
                redo=redo,
            )
        return corpus
