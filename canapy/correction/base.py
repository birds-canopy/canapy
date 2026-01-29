# Author: Nathan Trouvain at 07/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import toml
import copy

from pathlib import Path
from typing import Union, List, Dict

import attr


def correct_classes(corpus, corrections):
    df = corpus.dataset.copy()
    df = df.replace(to_replace={"label": corrections})
    return corpus.clone_with_df(df)


def correct_annots(corpus, corrections):
    df = corpus.dataset.copy()

    for index, corr in corrections.items():
        df.at[int(index), "label"] = corr

    silence_tag = corpus.config.transforms.annots.silence_tag
    df = df[df["label"] != silence_tag]

    return corpus.clone_with_df(df)


@attr.define()
class Corrector:
    checkpoint_directory: Union[Path, str] = attr.field(converter=Path)
    correction_history: List[Dict] = attr.field(default=list())

    @classmethod
    def from_checkpoints(cls, checkpoint_directory):
        ckpt_dir = Path(checkpoint_directory)

        correction_history = []
        for ckpt in sorted(ckpt_dir.glob("*.toml")):
            with open(ckpt, "r") as fp:
                correction = toml.load(fp)

            if "class" not in correction or "annot" not in correction:
                raise KeyError(
                    f"Unknown correction file format: {ckpt}. "
                    f"Should have 'class' and 'annot' keys."
                )

            # key should be an integer index, but TOML parses it to str
            correction["annot"] = {int(k): v for k, v in correction["annot"].items()}

            correction_history.append(correction)

        return cls(
            checkpoint_directory=checkpoint_directory,
            correction_history=correction_history,
        )

    def correct(
        self, corpus, class_corrections=None, annot_corrections=None, checkpoint=False
    ):
        corrections = {"class": class_corrections, "annot": annot_corrections}

        new_corpus = corpus
        if annot_corrections is not None:
            new_corpus = correct_annots(new_corpus, annot_corrections)

        if class_corrections is not None:
            new_corpus = correct_classes(new_corpus, class_corrections)

        if checkpoint:
            self.checkpoint(corrections)

        return new_corpus

    def correct_from_history(self, corpus, checkpoint_step):
        if checkpoint_step > len(self.correction_history):
            raise ValueError(
                f"Checkpoint step {checkpoint_step} can't be found in Corrector "
                f"history. Maximum checkpoint step: {len(self.correction_history)}."
            )

        corrections = self.correction_history[checkpoint_step]

        return self.correct(
            corpus,
            class_corrections=corrections["class"],
            annot_corrections=corrections["annot"],
        )

    def checkpoint(self, corrections):
        self.correction_history.append(corrections)

        cktp_step = str(len(self.correction_history))
        ckpt_correction_file = self.checkpoint_directory / ("correction-" + cktp_step)
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)

        corr = copy.deepcopy(corrections)
        with open(ckpt_correction_file, "w+") as fp:
            # TOML keys can't be integers. Convert the corrected indexes to str.
            corr["annot"] = {str(k): v for k, v in corr["annot"].items()}
            toml.dump(corr, fp)

        return self
