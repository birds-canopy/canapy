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


def plan_bounds(df, corrections, time_precision):
    """What a bounds correction does, before anything is written.

    Returns ``(moves, dropped)``: where each annotation ends up, and which ones
    the correction removes, both keyed by the annotation's key *before* the
    change. Keys rather than row positions, so the same plan can be replayed on
    another table holding the same annotations — the misclassified segments the
    eval page displays, for instance, which are a copy and not a view.
    """
    working = df.copy()
    unresolved, moved = [], []
    for key, (onset_s, offset_s) in corrections.items():
        matches = match_annotation(working, key, time_precision)
        if len(matches) == 0:
            unresolved.append(key)
            continue
        working.loc[matches, "onset_s"] = float(onset_s)
        working.loc[matches, "offset_s"] = float(offset_s)
        moved.extend(matches)

    if unresolved:
        logger.warning(
            f"{len(unresolved)} bounds correction(s) skipped: no matching "
            f"segment in the current corpus. Keys: {unresolved}"
        )

    moves = {
        make_annot_key(df.at[i, "notated_path"], df.at[i, "onset_s"]):
            (float(working.at[i, "onset_s"]), float(working.at[i, "offset_s"]))
        for i in moved
    }
    dropped = []

    # a correction wins over the annotations it grows into: writing the overlap
    # as asked would make it invisible instead, canapy giving the shared frames
    # to whichever annotation starts later
    for index in moved:
        onset = working.at[index, "onset_s"]
        offset = working.at[index, "offset_s"]
        others = working.index[
            (working["notated_path"] == working.at[index, "notated_path"])
            & (~working.index.isin(moved))
            & (working["onset_s"] < offset)
            & (working["offset_s"] > onset)
        ]
        for other in others:
            key = make_annot_key(df.at[other, "notated_path"], df.at[other, "onset_s"])
            if key in moves or key in dropped:
                continue
            covered_start = working.at[other, "onset_s"] >= onset
            covered_end = working.at[other, "offset_s"] <= offset
            if covered_start and covered_end:
                dropped.append(key)
            elif covered_start:
                moves[key] = (offset, float(working.at[other, "offset_s"]))
                working.at[other, "onset_s"] = offset
            else:
                moves[key] = (float(working.at[other, "onset_s"]), onset)
                working.at[other, "offset_s"] = onset

    if dropped:
        logger.warning(
            f"{len(dropped)} annotation(s) removed: fully covered by a segment "
            f"whose bounds were corrected."
        )

    return moves, dropped


def apply_bounds_plan(df, moves, dropped, time_precision):
    """Write a plan from ``plan_bounds`` onto any table of annotations."""
    out = df.copy()
    # every key is resolved against the table as it stands, before a single
    # write: a moved annotation can land on the onset of one being dropped, and
    # would then answer to its key
    targets = [
        (match_annotation(out, key, time_precision), bounds)
        for key, bounds in moves.items()
    ]
    gone = []
    for key in dropped:
        gone.extend(match_annotation(out, key, time_precision))

    for matches, (onset_s, offset_s) in targets:
        out.loc[matches, "onset_s"] = onset_s
        out.loc[matches, "offset_s"] = offset_s
    if gone:
        out = out.drop(index=gone)

    sort_keys = [k for k in ("annotation", "sequence", "onset_s") if k in out.columns]
    return out.sort_values(by=sort_keys, ignore_index=True)


def correct_bounds(corpus, corrections):
    """Move the onset and offset of annotations, leaving their labels alone.

    ``corrections`` maps an annotation key to its new ``(onset_s, offset_s)``.
    Applied after the label channels, never before: a key is built from the
    onset (see ``make_annot_key``), so moving one invalidates the key every
    other correction of the same annotation was captured under.

    A correction always wins over the annotations it grows into — see
    ``plan_bounds``.
    """
    time_precision = corpus.config.transforms.annots.time_precision
    moves, dropped = plan_bounds(corpus.dataset, corrections, time_precision)
    return corpus.clone_with_df(
        apply_bounds_plan(corpus.dataset, moves, dropped, time_precision)
    )


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
            # 'bounds' postdates the other two: checkpoints written before it
            # have no such table and must still load.
            correction["bounds"] = {
                key: (float(pair[0]), float(pair[1]))
                for key, pair in dict(correction.get("bounds", {})).items()
            }

            correction_history.append(correction)

        return cls(
            checkpoint_directory=checkpoint_directory,
            correction_history=correction_history,
        )

    def correct(
        self, corpus, class_corrections=None, annot_corrections=None,
        bounds_corrections=None, checkpoint=False
    ):
        corrections = {
            "class": class_corrections,
            "annot": annot_corrections,
            "bounds": bounds_corrections,
        }

        new_corpus = corpus
        if annot_corrections is not None:
            new_corpus = correct_annots(new_corpus, annot_corrections)

        if class_corrections is not None:
            new_corpus = correct_classes(new_corpus, class_corrections)

        # last: moving an onset changes the key the others were captured under
        if bounds_corrections is not None:
            new_corpus = correct_bounds(new_corpus, bounds_corrections)

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
            bounds_corrections=corrections.get("bounds"),
        )

    def checkpoint(self, corrections):
        self.correction_history.append(corrections)

        ckpt_step = str(len(self.correction_history))
        ckpt_correction_file = self.checkpoint_directory / ("correction-" + ckpt_step + ".toml")
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)

        corr = copy.deepcopy(corrections)
        with open(ckpt_correction_file, "w+") as fp:
            # every channel is written, empty ones included: toml.dump drops a
            # None, and from_checkpoints requires the tables to be there.
            corr["class"] = dict(corr.get("class") or {})
            # Annotation keys are already stable strings (see make_annot_key);
            # str() keeps backward compatibility with any legacy integer keys.
            corr["annot"] = {str(k): v for k, v in (corr.get("annot") or {}).items()}
            corr["bounds"] = {
                str(k): [float(v[0]), float(v[1])]
                for k, v in (corr.get("bounds") or {}).items()
            }
            toml.dump(corr, fp)

        return self
