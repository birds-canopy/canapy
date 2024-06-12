# Author: Nathan Trouvain at 04/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
"""
Echo State Network related functions.
"""

import numpy as np
import reservoirpy as rpy

from reservoirpy.nodes import Reservoir, Ridge, ESN
from reservoirpy.mat_gen import fast_spectral_initialization

from canapy.utils.exceptions import NotTrainedError
from .mfccs import load_mfccs_for_annotation


def maximum_a_posteriori(logits, classes=None):
    """Select the neuron index with maximum prediction probability.

    Parameters
    ----------
    logits : np.ndarray
        Model predictions, array of shape (samples, classes)
    classes : list of str, optional
        Class labels.

    Returns
    -------
    np.ndarray
        Array of shape (samples, ) with class predictions
        (as class labels or class index if classes=None)
    """
    logits = np.atleast_2d(logits)

    predictions = np.argmax(logits, axis=1)

    if classes is not None:
        predictions = np.take(classes, predictions)

    return predictions


def init_esn_model(model_config, input_dim, audio_features, seed=None):
    """Initialize an Echo State Network for MFCC frame classification.

    Parameters
    ----------
    model_config : Config
        A Config object holding model parameters.
    input_dim : int
        MFCC dimension (number of coefficients)
    audio_features : list of str
        Audio features used as input to the model. Must be
        a list containing "mfcc", "delta", "delta2", or
        any of their combinations.
    seed : int, optional
        Random state seed.

    Returns
    -------
    reservoirpy.nodes.ESN
        A reservoirpy ESN model.
    """
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
        W=fast_spectral_initialization,
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
    """Produce annotations with an ESN-based annotator.
    Annotator must be trained.

    Parameters
    ----------
    annotator : Annotator
        An ESN-based annotator.
    corpus : Corpus
        A corpus to annotate.
    return_raw : bool, default to False
        If True, returns the model raw outputs
        (probability scores of each output neurons)
        along with the annotated segments.
    redo_transforms : bool, default to False
        If True, preprocessing is reapplied.

    Returns
    -------
    list of str, list of np.ndarray, list of np.ndarray
        Audio file names, corresponding annotations (sequences of labels),
        and model raw predictions (if raw_preds=True, else return None)
    """
    if not annotator.trained:
        raise NotTrainedError(
            "Call .fit on annotated data (Corpus) before calling .predict."
        )

    corpus = annotator.transforms(
        corpus,
        purpose="annotation",
        output_directory=corpus.spec_directory,
        redo=redo_transforms,
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
