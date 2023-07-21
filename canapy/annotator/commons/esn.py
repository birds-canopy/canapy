# Author: Nathan Trouvain at 04/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import numpy as np
import reservoirpy as rpy

from reservoirpy.nodes import Reservoir, Ridge, ESN
from reservoirpy.mat_gen import fast_spectral_initialization

from canapy.utils.exceptions import NotTrainedError
from .mfccs import load_mfccs_for_annotation


def maximum_a_posteriori(logits, classes=None):
    logits = np.atleast_2d(logits)

    predictions = np.argmax(logits, axis=1)

    if classes is not None:
        predictions = np.take(classes, predictions)

    return predictions


def init_esn_model(model_config, input_dim, audio_features, seed=None):

    rpy.set_seed(seed)

    scalings = []
    if "mfcc" in audio_features:
        iss = np.ones((input_dim,)) * model_config.iss
        scalings.append(iss)
    if "delta" in audio_features:
        isd = np.ones((input_dim,)) * model_config.isd
        scalings.append(isd)
    if "delta2" in audio_features:
        isd2 = np.ones((input_dim,)) * model_config.isd2
        scalings.append(isd2)

    input_scaling = np.concatenate(scalings, axis=0)
    bias_scaling = model_config.iss

    reservoir = Reservoir(
        model_config.units,
        sr=model_config.sr,
        lr=model_config.leak,
        input_scaling=input_scaling,
        bias_scaling=bias_scaling,
        W=fast_spectral_initialization
    )

    readout = Ridge(ridge=model_config.ridge)

    return ESN(
        reservoir=reservoir,
        readout=readout,
        workers=model_config.workers,
        backend=model_config.backend,
    )


def predict_with_esn(
    annotator,
    corpus,
    return_raw=False,
    redo_transforms=False,
):
    if not annotator.trained:
        raise NotTrainedError(
            "Call .fit on annotated data (Corpus) before calling " ".predict."
        )

    corpus = annotator.transforms(
        corpus,
        purpose="annotation",
        output_directory=annotator.spec_directory,
    )

    notated_paths, mfccs = load_mfccs_for_annotation(corpus)

    raw_preds = annotator.rpy_model.run(mfccs)

    if isinstance(raw_preds, np.ndarray) and raw_preds.ndim < 3:
        raw_preds = [raw_preds]

    cls_preds = []
    for y in raw_preds:
        y_map = maximum_a_posteriori(y, classes=annotator.vocab)
        cls_preds.append(y_map)

    if not return_raw:
        raw_preds = None

    return notated_paths, cls_preds, raw_preds
