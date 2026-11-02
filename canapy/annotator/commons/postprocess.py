# Author: Nathan Trouvain at 05/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: BSD-3-Clause
# Copyright: Nathan Trouvain
import numpy as np
import pandas as pd

from ...corpus import Corpus
from ...timings import frames_to_seconds, frames_to_timed_df, seconds_to_audio
from ...utils.arrays import to_structured


def extract_vocab(corpus, silence_tag):
    vocab = corpus.dataset["label"].unique().tolist()
    if silence_tag not in vocab:
        vocab += [silence_tag]
    return sorted(vocab)


def _runs(labels):
    """Runs of identical labels, as (start, exclusive end, label)."""
    labels = np.asarray(labels, dtype=object)
    if not len(labels):
        return []
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts, ends = np.r_[0, changes], np.r_[changes, len(labels)]
    return [(int(s), int(e), labels[s]) for s, e in zip(starts, ends)]


def _duration_to_frames(duration, hop_length):
    return max(1, int(np.ceil(duration / hop_length - 1e-6)))


def _tiebreak_ranks(labels):
    """Ranks that ignore label names: most frequent in the file, then first seen."""
    unique, first, counts = np.unique(
        np.asarray(labels, dtype=str), return_index=True, return_counts=True
    )
    order = np.lexsort((first, -counts))
    return {unique[i]: rank for rank, i in enumerate(order)}


def _zone_winner(unique, counts, neighbours, ranks):
    """Most present label; a tie goes to a neighbour if one is tied, else to the ranks."""
    tied = [label for label, count in zip(unique, counts) if count == counts.max()]
    for neighbour in neighbours:
        if neighbour in tied:
            return neighbour
    return min(tied, key=lambda label: ranks[str(label)])


def _best_cut(zone, left, right, towards_left=True):
    """Cut c maximising #(zone[:c] == left) + #(zone[c:] == right).

    A tie goes to the middle, then to the side of the longest neighbour.
    """
    left_score = np.r_[0, np.cumsum(zone == left)]
    right_score = np.r_[np.cumsum((zone == right)[::-1])[::-1], 0]
    score = left_score + right_score
    best = np.flatnonzero(score == score.max())
    distance = np.abs(best - len(zone) / 2)
    best = best[distance == distance.min()]
    return int(best.max() if towards_left else best.min())


def _resolve_short_runs(labels, min_frames, min_silence_frames, silence_tag, zone_factor):
    """Share out the zones of consecutive short runs between their long neighbours.

    Long runs are never modified. A zone becomes its own segment only if a label
    other than its neighbours fills ``zone_factor`` times the minimum duration.

    Returns the labels per frame and a run id per frame.
    """
    labels = np.asarray(labels, dtype=object).copy()
    n = len(labels)
    runs = _runs(labels)
    is_long = [
        (end - start) >= (min_silence_frames if label == silence_tag else min_frames)
        for start, end, label in runs
    ]
    ranks = _tiebreak_ranks(labels)
    ids = np.zeros(n, dtype=int)
    for i, (start, end, _) in enumerate(runs):
        ids[start:end] = i

    i = 0
    while i < len(runs):
        if is_long[i]:
            i += 1
            continue
        j = i
        while j < len(runs) and not is_long[j]:
            j += 1
        zone_start, zone_end = runs[i][0], runs[j - 1][1]
        zone = labels[zone_start:zone_end]
        left = runs[i - 1][2] if i > 0 else silence_tag
        right = runs[j][2] if j < len(runs) else silence_tag
        left_id = ids[zone_start - 1] if i > 0 else -1
        right_id = ids[zone_end] if j < len(runs) else -2

        unique, counts = np.unique(zone, return_counts=True)
        winner = _zone_winner(unique, counts, (left, right), ranks)
        threshold = zone_factor * (
            min_silence_frames if winner == silence_tag else min_frames
        )
        if counts[list(unique).index(winner)] >= threshold and winner not in (left, right):
            labels[zone_start:zone_end] = winner
            ids[zone_start:zone_end] = runs[i][0] + n
        else:
            left_length = runs[i - 1][1] - runs[i - 1][0] if i > 0 else 0
            right_length = runs[j][1] - runs[j][0] if j < len(runs) else 0
            cut = _best_cut(zone, left, right, left_length >= right_length)
            labels[zone_start:zone_start + cut] = left
            labels[zone_start + cut:zone_end] = right
            ids[zone_start:zone_start + cut] = left_id
            ids[zone_start + cut:zone_end] = right_id
        i = j
    return labels, ids


def _silence_short_head(labels, min_frames, silence_tag, head_factor):
    """Silence the first segments while they are both short and at the start of the file."""
    labels = labels.copy()
    if not head_factor:
        return labels
    threshold = head_factor * min_frames
    while True:
        segments = [run for run in _runs(labels) if run[2] != silence_tag]
        if (
            not segments
            or segments[0][0] >= threshold
            or segments[0][1] - segments[0][0] >= threshold
        ):
            return labels
        start, end, _ = segments[0]
        labels[start:end] = silence_tag


def frame_labels_to_annots(
    frame_predictions,
    notated_path,
    hop_length,
    sampling_rate,
    time_precision,
    min_label_duration,
    min_silence_duration,
    min_silence_gap,
    silence_tag="SIL",
    lonely_labels=None,
    zone_factor=2.0,
    head_factor=3.0,
):
    """Labels per frame to annotations (label, onset_s, offset_s, notated_path).

    Durations are counted in whole frames, so that a threshold does not depend
    on how the bounds of a run round to ``time_precision``. Silences are left
    out of the result.
    """
    empty = pd.DataFrame(
        {
            "label": pd.Series(dtype=object),
            "onset_s": pd.Series(dtype=float),
            "offset_s": pd.Series(dtype=float),
            "notated_path": pd.Series(dtype=object),
        }
    )
    labels = np.asarray(frame_predictions, dtype=object)
    n = len(labels)
    if n == 0:
        return empty

    lonely_labels = set(lonely_labels or [])
    min_frames = _duration_to_frames(min_label_duration, hop_length)
    min_silence_frames = _duration_to_frames(min_silence_duration, hop_length)

    labels, ids = _resolve_short_runs(
        labels, min_frames, min_silence_frames, silence_tag, zone_factor
    )
    labels = _silence_short_head(labels, min_frames, silence_tag, head_factor)

    # One run is one segment; lonely labels stay apart even across a resolved zone
    ids = np.where(np.isin(labels, list(lonely_labels)), ids, -1)
    keys = np.array([f"{l}\x00{i}" for l, i in zip(labels, ids)], dtype=object)
    segments = []
    for start, end, _ in _runs(keys):
        label = labels[start]
        if label == silence_tag:
            continue
        if (
            segments
            and segments[-1][2] == label
            and label not in lonely_labels
            and (start - segments[-1][1]) * hop_length <= min_silence_gap + 1e-9
        ):
            segments[-1][1] = end
        else:
            segments.append([start, end, label])
    if not segments:
        return empty

    frame_size = seconds_to_audio(hop_length, sampling_rate)
    onsets, offsets = frames_to_seconds(
        np.arange(n), frame_size, sampling_rate, time_precision
    )
    return pd.DataFrame(
        {
            "label": [label for _, _, label in segments],
            "onset_s": np.array([onsets[s] for s, _, _ in segments], dtype=float),
            "offset_s": np.array([offsets[e - 1] for _, e, _ in segments], dtype=float),
            "notated_path": [notated_path] * len(segments),
        }
    )


def frames_to_annots(frame_predictions, notated_path, config):
    """``frame_labels_to_annots`` with its parameters read from a config."""
    annots, audio = config.transforms.annots, config.transforms.audio
    return frame_labels_to_annots(
        frame_predictions,
        notated_path=notated_path,
        hop_length=float(audio.hop_length),
        sampling_rate=audio.sampling_rate,
        time_precision=float(annots.time_precision),
        min_label_duration=float(annots.min_label_duration),
        min_silence_duration=float(
            annots.get("min_silence_duration", annots.min_label_duration)
        ),
        min_silence_gap=float(annots.min_silence_gap),
        silence_tag=annots.silence_tag,
        lonely_labels=list(annots.lonely_labels or []),
        zone_factor=float(annots.get("zone_factor", 2.0)),
        head_factor=float(annots.get("head_factor", 3.0)),
    )


def predictions_to_corpus(
    notated_paths,
    cls_preds,
    frame_size,
    sampling_rate,
    time_precision,
    min_label_duration,
    min_silence_gap,
    silence_tag,
    lonely_labels,
    config=None,
    raw_preds=None,
):
    frame_dfs = []
    annot_dfs = []
    for y_pred, notated_path in zip(cls_preds, notated_paths):
        frames = frames_to_timed_df(
            y_pred,
            notated_path=notated_path,
            frame_size=frame_size,
            sampling_rate=sampling_rate,
            time_precision=time_precision,
        )

        if config is not None:
            annots = frames_to_annots(y_pred, notated_path, config)
        else:
            annots = frame_labels_to_annots(
                y_pred,
                notated_path=notated_path,
                hop_length=frame_size / sampling_rate,
                sampling_rate=sampling_rate,
                time_precision=time_precision,
                min_label_duration=min_label_duration,
                min_silence_duration=min_label_duration,
                min_silence_gap=min_silence_gap,
                silence_tag=silence_tag,
                lonely_labels=lonely_labels,
            )

        frame_dfs.append(frames)
        annot_dfs.append(annots)

    frame_df = pd.concat(frame_dfs)
    annots_df = pd.concat(annot_dfs)

    corpus = Corpus.from_df(annots_df, config=config)

    corpus.register_data_resource("frames_predictions", frame_df)

    if raw_preds is not None:
        if not isinstance(raw_preds, dict):
            raw_preds = dict(zip(notated_paths, raw_preds))
        # Save raw neural network outputs (logits, softmax activity...) as
        # structured array for convenience. This allows for easy memory mapping to avoid
        # memory issues.
        corpus.register_data_resource("nn_output", to_structured(raw_preds))

    return corpus
