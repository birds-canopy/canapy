# Licence: BSD-3-Clause
"""Moving a whole corpus from phrase-level to syllable-level annotation."""
import logging

import pandas as pd
import soundfile as sf

from . import segmenter
from .features import contours
from .grouping import estimate, pool_by_label
from ..utils.exceptions import AlreadySegmented

logger = logging.getLogger("canapy")

SORT_KEYS = ("annotation", "sequence", "onset_s")

POOL_PERIOD = True

# resource marking a corpus this module has produced
SEGMENTED_MARKER = "syllable_segmented"
# longest `min_label_duration` that only a syllable-level corpus reaches; the
# phrase-level default is 0.02
SYLLABLE_SCALE = 0.01


def segmentation_params(config):
    """Segmentation parameters, falling back to the module defaults.

    Every key is an argument of ``segment_signal``; callers spread this dict
    straight into it. ``pool_period`` is not one, hence ``pools_period``.
    """
    section = getattr(config, "segmentation", None) or {}

    wiener_max = section.get("wiener_max", segmenter.WIENER_MAX)
    band = section.get("band", segmenter.BAND)

    annots = getattr(config, "transforms", None)
    annots = getattr(annots, "annots", None) if annots is not None else None

    return dict(
        min_syllable_duration=section.get(
            "min_syllable_duration", segmenter.MIN_SYLLABLE_DURATION
        ),
        min_silence_gap=getattr(annots, "min_silence_gap", segmenter.MIN_SILENCE_GAP),
        # "otsu" reads the gate off the flatness contour instead of imposing it
        wiener_max=None if wiener_max == "otsu" else wiener_max,
        band=(float(band[0]), float(band[1])),
        hop=section.get("hop", segmenter.HOP),
        eta2_min=section.get("eta2_min", segmenter.ETA2_MIN),
        group_syllables=bool(section.get("group_syllables",
                                         segmenter.GROUP_SYLLABLES)),
    )


def pools_period(config):
    """Whether the repetition period is estimated per label or per phrase.

    The period of a repeated unit is a property of the label rather than of one
    phrase, so pooling takes the median over every phrase carrying that label,
    which resists the octave errors a single phrase can make. Turning it off
    gives each phrase the period it measured alone.
    """
    section = getattr(config, "segmentation", None) or {}
    return bool(section.get("pool_period", POOL_PERIOD))


def untouched_labels(config):
    """Labels a syllable segmenter has no business splitting.

    The silence tag, plus `lonely_labels` — TRASH is a bin, and a call is a
    single vocalisation, neither of which is a repeated phrase.
    """
    annots = getattr(config, "transforms", None)
    annots = getattr(annots, "annots", None) if annots is not None else None
    labels = {getattr(annots, "silence_tag", "SIL")}
    labels.update(str(label) for label in getattr(annots, "lonely_labels", []) or [])
    return labels


def _pool_periods(corpus, params, untouched):
    """First pass: what each phrase says about its label, pooled per label.

    Costs a second read of the audio and a second STFT; keeping the envelopes
    would avoid it, at the price of holding the whole corpus in memory at a
    0.5 ms hop.
    """
    estimates = {}
    for path, phrases in corpus.dataset.groupby("notated_path", sort=False):
        try:
            signal, sr = _read_mono(path)
        except Exception:
            continue

        for entry in phrases.to_dict("records"):
            if str(entry["label"]) in untouched:
                continue
            onset, offset = float(entry["onset_s"]), float(entry["offset_s"])
            excerpt = signal[int(onset * sr): int(offset * sr)]
            segments, _ = segmenter.segment_signal(
                excerpt, sr, **{**params, "group_syllables": False})
            if len(segments) < 2:
                continue
            times, energy, _, _ = contours(excerpt, sr, band=params["band"],
                                           hop=params["hop"])
            measured = estimate(segments, times, energy)
            if measured:
                estimates.setdefault(str(entry["label"]), []).append(measured)

    pooled = pool_by_label(estimates)
    for label, values in sorted(pooled.items()):
        logger.debug(f"{label}: period {values['period'] * 1000:.0f} ms over "
                     f"{values['n_phrases']} phrases")
    return pooled


def _read_mono(path):
    signal, sr = sf.read(str(path))
    if signal.ndim > 1:
        signal = signal[:, 0]
    return signal, sr


def _check_not_segmented(corpus, annots):
    """Refuse a corpus already annotated at syllable level.

    Two signals. The marker is exact but only survives within a session; a
    corpus loaded from disk, or predicted by a model trained on syllables, is
    caught by ``min_label_duration`` having been pushed to syllable scale.

    That second test is against ``SYLLABLE_SCALE`` and not against the
    configured ``min_syllable_duration``: the latter is a tunable the
    segmentation panel exposes, and comparing to it refused a plain
    phrase-level corpus as soon as it was raised.
    """
    if SEGMENTED_MARKER in corpus.data_resources:
        raise AlreadySegmented(
            "This corpus has already been segmented into syllables. Reload it "
            "to start over from phrase-level annotations."
        )

    min_label_duration = getattr(annots, "min_label_duration", None)
    if min_label_duration is not None and min_label_duration <= SYLLABLE_SCALE:
        raise AlreadySegmented(
            f"min_label_duration ({min_label_duration}s) is at syllable scale "
            f"(<= {SYLLABLE_SCALE}s), which a corpus only reaches once it has "
            f"been segmented. Reload it to start over from phrase-level "
            f"annotations."
        )


def to_syllable_level(corpus):
    """Replace every phrase annotation with the syllables it contains.

    Each phrase row becomes N rows carrying the same label, bounded by the
    phrase it came from. Inter-syllable silences are left implicit, as
    inter-phrase silences already are. The operation is one-way: Otsu tightens
    the phrase bounds, so merging the syllables back does not restore them.

    Labels listed by ``untouched_labels`` — the silence tag, TRASH, calls —
    are copied over untouched.

    Raises AlreadySegmented if the corpus has already been through it: Otsu
    assumes an alternation of song and silence, which an isolated syllable does
    not offer.
    """
    params = segmentation_params(corpus.config)
    pool_period = pools_period(corpus.config)
    annots = getattr(corpus.config, "transforms", None)
    annots = getattr(annots, "annots", None) if annots is not None else None
    untouched = untouched_labels(corpus.config)

    _check_not_segmented(corpus, annots)

    # without pooling, `group_segments` falls back to its own estimate
    pooled = _pool_periods(corpus, params, untouched) \
        if params["group_syllables"] and pool_period else {}

    rows = []
    n_phrases = n_refused = 0

    for path, phrases in corpus.dataset.groupby("notated_path", sort=False):
        try:
            signal, sr = _read_mono(path)
        except Exception as error:
            logger.error(f"Could not read {path}, left unsegmented: {error}")
            rows.extend(phrases.to_dict("records"))
            continue

        for entry in phrases.to_dict("records"):
            if str(entry["label"]) in untouched:
                rows.append(entry)
                continue

            n_phrases += 1
            onset, offset = float(entry["onset_s"]), float(entry["offset_s"])
            excerpt = signal[int(onset * sr): int(offset * sr)]
            label = pooled.get(str(entry["label"]), {})
            segments, eta2 = segmenter.segment_signal(
                excerpt, sr, **params, period=label.get("period"))

            if not segments:
                n_refused += 1
                logger.debug(
                    f"{path} [{onset:.3f}, {offset:.3f}] left unsegmented "
                    f"(eta2={eta2:.2f})"
                )
                rows.append(entry)
                continue

            # the STFT pads, so the last segment can reach a fraction of a
            # millisecond past the phrase it came from
            rows.extend(
                {**entry,
                 "onset_s": min(onset + start, offset),
                 "offset_s": min(onset + end, offset)}
                for start, end in segments
            )

    dataset = pd.DataFrame(rows)
    dataset = dataset.sort_values(
        by=[key for key in SORT_KEYS if key in dataset.columns], ignore_index=True
    )

    logger.info(
        f"Syllable segmentation: {n_phrases} phrases -> {len(dataset)} annotations "
        f"({n_refused} left unsegmented)"
    )
    segmented = corpus.clone_with_df(dataset)
    segmented.register_data_resource(SEGMENTED_MARKER, True)
    return segmented
