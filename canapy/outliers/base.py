# Licence: BSD-3-Clause
"""Annotated segments whose label does not match their neighbourhood.

Each segment (silences excepted) is described by its mean MFCC over its frames
(the static coefficients, standardized), and this description is embedded in a
low dimensional space by UMAP. A segment is an outlier when

- none of its ``neighbors`` nearest segments has its label,
- the nearest segment of its label is further than for ``percentile`` % of the
  segments,
- at least ``min_other`` of its neighbors share another label (the label it
  sounds like).

Whole files can also be mislabeled, their annotations shifted in time: most of
their segments then sound like another label. For each file where less than
``min_file_agreement`` of the segments sound like their label, the shift of its
annotations maximizing the agreement of its labels with a kNN of the segments of
the other files is estimated.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ..correction import make_annot_key
from ..timings import seconds_to_audio, seconds_to_frames
from ..transforms.commons.audio import compute_mfcc

logger = logging.getLogger("canapy")

NEIGHBORS = 10
MIN_OTHER = 8
PERCENTILE = 99.0
N_COMPONENTS = 3
MIN_FILE_AGREEMENT = 0.5
MAX_SHIFT = 2.0
SHIFT_STEP = 0.05

# resource of the corpus holding the MFCC of each file (see compute_mfcc)
MFCC_RESOURCE = "syn_mfcc"


def outliers_params(config):
    """Outlier detection parameters, falling back to the module defaults."""
    section = getattr(config, "outliers", None) or {}
    return dict(
        neighbors=int(section.get("neighbors", NEIGHBORS)),
        min_other=int(section.get("min_other", MIN_OTHER)),
        percentile=float(section.get("percentile", PERCENTILE)),
        n_components=int(section.get("n_components", N_COMPONENTS)),
        min_file_agreement=float(
            section.get("min_file_agreement", MIN_FILE_AGREEMENT)
        ),
        max_shift=float(section.get("max_shift", MAX_SHIFT)),
        shift_step=float(section.get("shift_step", SHIFT_STEP)),
    )


def _segments(corpus):
    """The annotated segments of the corpus, silences excepted."""
    silence_tag = corpus.config.transforms.annots.silence_tag
    df = corpus.dataset
    return df[df["label"] != silence_tag]


def _file_features(corpus):
    """{notated_path: static MFCC (n_mfcc, n_frames)} of the files of the corpus.

    The MFCC are those used to train the annotators (``syn_mfcc`` resource),
    computed and stored in the spectrogram directory of the corpus if missing.
    """
    if MFCC_RESOURCE not in corpus.data_resources:
        corpus = compute_mfcc(
            corpus, output_directory=None, resource_name=MFCC_RESOURCE
        )
    registry = corpus.data_resources[MFCC_RESOURCE]
    feature_paths = dict(zip(registry["notated_path"].astype(str),
                             registry["feature_path"]))

    audio_config = corpus.config.transforms.audio
    # static coefficients only: the deltas, stacked after them, average to ~0
    n_static = audio_config.n_mfcc if "mfcc" in audio_config.audio_features else None

    features = {}
    for notated_path in _segments(corpus)["notated_path"].unique():
        feature_path = Path(feature_paths[str(notated_path)])
        if not feature_path.exists():  # (np.savez appends .npz to other extensions)
            feature_path = feature_path.with_name(feature_path.name + ".npz")
        archive = np.load(str(feature_path))
        feature = archive["feature"].squeeze()
        features[notated_path] = feature[:n_static] if n_static else feature
    return features


def _mean_features(segments, features, config, shift=0.0):
    """Mean of the features over the frames of each segment (n_segments, n_features).

    The annotation times are shifted by ``shift`` seconds (kept in the file).
    """
    sampling_rate = config.transforms.audio.sampling_rate
    hop = seconds_to_audio(config.transforms.audio.hop_length, sampling_rate)
    onsets = seconds_to_frames(segments["onset_s"].to_numpy() + shift, hop, sampling_rate)
    offsets = seconds_to_frames(segments["offset_s"].to_numpy() + shift, hop, sampling_rate)

    means = []
    for path, onset, offset in zip(segments["notated_path"], onsets, offsets):
        feature = features[path]
        n_frames = feature.shape[1]
        start = int(np.clip(onset, 0, n_frames - 1))
        end = int(np.clip(offset, start + 1, n_frames))
        means.append(feature[:, start:end].mean(axis=1))
    return np.array(means)


def _embed(X, n_components, seed):
    """UMAP embedding of the rows of X."""
    import umap  # (imported here: slow to import)

    # the layout of umap-learn 0.6 (compatibility_layout=False): reproducible with a seed
    # while running in parallel
    return umap.UMAP(
        n_components=n_components,
        n_neighbors=min(15, len(X) - 1),
        min_dist=0.1,
        random_state=seed,
        compatibility_layout=False,
        n_jobs=-1,
    ).fit_transform(X)


def _flag(Y, labels, neighbors, min_other, percentile):
    """(outlier flags, label most of the neighbors share, the nearest neighbors) in the embedding."""
    nearest = NearestNeighbors(n_neighbors=neighbors + 1).fit(Y).kneighbors(Y)[1][:, 1:]
    neighbor_labels = labels[nearest]

    # distance to the nearest segment of the same label (inf for a lone label)
    to_same = np.full(len(Y), np.inf)
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        if len(idx) > 1:
            to_same[idx] = NearestNeighbors(n_neighbors=2).fit(Y[idx]).kneighbors(Y[idx])[0][:, 1]
    finite = to_same[np.isfinite(to_same)]
    far = to_same > (np.percentile(finite, percentile) if len(finite) else np.inf)

    majority, majority_count = [], []
    for row in neighbor_labels:
        values, counts = np.unique(row, return_counts=True)
        majority.append(values[np.argmax(counts)])
        majority_count.append(counts.max())
    majority_count = np.array(majority_count)

    flags = ((neighbor_labels != labels[:, None]).all(axis=1)
             & far
             & (majority_count >= min_other))
    return flags, np.array(majority), nearest


def find_outliers(corpus, params=None, seed=None):
    """Segments whose label does not match their neighbourhood (see the module docstring).

    Parameters
    ----------
    corpus : Corpus
        Annotated corpus, with its audio or its MFCC.
    params : dict, optional
        Parameters, as ``outliers_params`` (default: those of the corpus config).
    seed : int, optional
        Seed of UMAP (default: ``misc.seed`` of the config).

    Returns
    -------
    pd.DataFrame
        One row per segment (silences excepted): ``key`` (``make_annot_key``),
        ``notated_path``, ``onset_s``, ``offset_s``, ``label``, ``sounds_like``
        (the label most of its neighbors share), ``neighbors_agreement``
        (fraction of its neighbors sharing its label), ``outlier``, and the
        coordinates ``umap_<i>`` of the embedding.
    """
    params = outliers_params(corpus.config) if params is None else params
    if seed is None:
        seed = corpus.config.misc.seed if "misc" in corpus.config else None

    segments = _segments(corpus)
    neighbors = min(params["neighbors"], len(segments) - 1)
    if neighbors < 1:
        raise ValueError("At least two annotated segments are needed.")

    X = StandardScaler().fit_transform(
        _mean_features(segments, _file_features(corpus), corpus.config)
    )
    labels = segments["label"].to_numpy().astype(str)

    result = pd.DataFrame({
        "key": [make_annot_key(p, t) for p, t in zip(segments["notated_path"], segments["onset_s"])],
        "notated_path": segments["notated_path"].to_numpy(),
        "onset_s": segments["onset_s"].to_numpy(),
        "offset_s": segments["offset_s"].to_numpy(),
        "label": labels,
    })

    logger.info(f"Outliers: UMAP of {len(segments)} segments...")
    Y = _embed(X, params["n_components"], seed)
    outlier, majority, nearest = _flag(
        Y, labels, neighbors, params["min_other"], params["percentile"]
    )
    result["sounds_like"] = majority
    result["neighbors_agreement"] = (labels[nearest] == labels[:, None]).mean(axis=1)
    result["outlier"] = outlier
    for i in range(Y.shape[1]):
        result[f"umap_{i}"] = Y[:, i]

    logger.info(f"Outliers: {outlier.sum()} among {len(segments)} segments.")
    return result


def find_shifted_files(corpus, outliers, params=None):
    """Files whose segments rarely sound like their label, and the estimated shift
    of their annotations (see the module docstring).

    Parameters
    ----------
    corpus : Corpus
        The corpus given to ``find_outliers``.
    outliers : pd.DataFrame
        Result of ``find_outliers``.
    params : dict, optional
        Parameters, as ``outliers_params`` (default: those of the corpus config).

    Returns
    -------
    pd.DataFrame
        One row per suspicious file: ``notated_path``, ``segments``,
        ``agreement`` (fraction of its segments sounding like their label),
        ``outliers``, ``knn_agreement`` (of its labels with a kNN of the other
        files), ``best_shift`` (s) and ``knn_agreement_after_shift``.
    """
    params = outliers_params(corpus.config) if params is None else params
    columns = ["notated_path", "segments", "agreement", "outliers", "knn_agreement",
               "best_shift", "knn_agreement_after_shift"]

    by_file = outliers.groupby("notated_path")
    agreement = (outliers["sounds_like"] == outliers["label"]).groupby(outliers["notated_path"]).mean()
    suspicious = agreement[agreement < params["min_file_agreement"]].index
    if len(suspicious) == 0 or len(suspicious) == by_file.ngroups:
        return pd.DataFrame(columns=columns)

    segments = _segments(corpus)
    features = _file_features(corpus)
    scaler = StandardScaler().fit(_mean_features(segments, features, corpus.config))
    others = segments[~segments["notated_path"].isin(suspicious)]
    knn = KNeighborsClassifier(n_neighbors=min(params["neighbors"], len(others))).fit(
        scaler.transform(_mean_features(others, features, corpus.config)),
        others["label"].astype(str),
    )

    step = params["shift_step"]
    shifts = np.round(np.arange(-params["max_shift"], params["max_shift"] + step / 2, step), 6)
    zero = int(np.argmin(np.abs(shifts)))
    rows = []
    for path in suspicious:
        file_segments = segments[segments["notated_path"] == path]
        labels = file_segments["label"].astype(str).to_numpy()
        accuracy = np.array([
            np.mean(knn.predict(scaler.transform(
                _mean_features(file_segments, features, corpus.config, shift))) == labels)
            for shift in shifts
        ])
        best = int(np.argmax(accuracy))
        rows.append(dict(
            notated_path=path,
            segments=len(file_segments),
            agreement=agreement[path],
            outliers=int(by_file.get_group(path)["outlier"].sum()),
            knn_agreement=accuracy[zero],
            best_shift=float(shifts[best]),
            knn_agreement_after_shift=accuracy[best],
        ))
    return pd.DataFrame(rows, columns=columns)
