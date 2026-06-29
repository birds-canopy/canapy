# Author: Nathan Trouvain at 07/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: BSD-3-Clause
# Copyright: Nathan Trouvain
import logging
import toml
import copy

from pathlib import Path
from typing import Union, List, Dict

import attr

logger = logging.getLogger("canapy")

# Separator used to build a stable, human-readable annotation key.
ANNOT_KEY_SEP = "::"


def make_annot_key(notated_path, onset_s):
    """Build a stable identifier for an annotation.

    Annotation corrections used to be keyed by the dataset's positional index
    (a ``RangeIndex``), but transforms such as ``merge_labels`` or
    ``remove_short_labels`` reset that index, so a key captured in one view
    could silently point to another row (or to nothing) once committed.

    ``(notated_path, onset_s)`` uniquely identifies a segment and survives
    re-indexing, which makes corrections robust to those transforms.
    """
    return f"{notated_path}{ANNOT_KEY_SEP}{float(onset_s):.6f}"


def match_annotation(df, key, time_precision=1e-3):
    """Return the index labels of rows in ``df`` matching a stable annotation key.

    Returns an empty index when the annotation no longer exists (e.g. it was
    merged into another segment), so callers can report it instead of crashing.
    Legacy/malformed keys (e.g. a positional index from an old checkpoint) also
    resolve to an empty index rather than raising.
    """
    key = str(key)
    if ANNOT_KEY_SEP not in key:
        return df.index[:0]
    notated_path, onset_str = key.rsplit(ANNOT_KEY_SEP, 1)
    try:
        onset_s = float(onset_str)
    except ValueError:
        return df.index[:0]
    atol = max(float(time_precision), 1e-6)
    mask = (df["notated_path"].astype(str) == notated_path) & (
        (df["onset_s"] - onset_s).abs() <= atol
    )
    return df.index[mask]


def correct_classes(corpus, corrections):
    df = corpus.dataset.copy()
    df = df.replace(to_replace={"label": corrections})
    return corpus.clone_with_df(df)


def correct_annots(corpus, corrections):
    df = corpus.dataset.copy()
    time_precision = corpus.config.transforms.annots.time_precision

    unresolved = []
    for key, corr in corrections.items():
        matches = match_annotation(df, key, time_precision)
        if len(matches) == 0:
            unresolved.append(key)
            continue
        df.loc[matches, "label"] = corr

    if unresolved:
        logger.warning(
            f"{len(unresolved)} annotation correction(s) skipped: no matching "
            f"segment in the current corpus (the annotation may have been merged "
            f"or removed). Keys: {unresolved}"
        )

    silence_tag = corpus.config.transforms.annots.silence_tag
    df = df[df["label"] != silence_tag]

    return corpus.clone_with_df(df)


@attr.define()
class Corrector:
    checkpoint_directory: Union[Path, str] = attr.field(converter=Path)
    correction_history: List[Dict] = attr.field(factory=list)

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

            # Annotation keys are stable string identifiers (see make_annot_key),
            # which TOML stores and loads back as-is. No int conversion.
            correction["annot"] = dict(correction["annot"])

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
        if checkpoint_step >= len(self.correction_history):
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

        ckpt_step = str(len(self.correction_history))
        ckpt_correction_file = self.checkpoint_directory / ("correction-" + ckpt_step + ".toml")
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)

        corr = copy.deepcopy(corrections)
        with open(ckpt_correction_file, "w+") as fp:
            # Annotation keys are already stable strings (see make_annot_key);
            # str() keeps backward compatibility with any legacy integer keys.
            if corr["annot"] is not None:
                corr["annot"] = {str(k): v for k, v in corr["annot"].items()}
            else:
                corr["annot"] = {}
            toml.dump(corr, fp)

        return self
