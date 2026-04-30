# Author: Nathan Trouvain at 04/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import numpy as np
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy import ESN
from reservoirpy.mat_gen import fast_spectral_initialization
from canapy.utils.exceptions import NotTrainedError
from .mfccs import load_mfccs_for_annotation

def maximum_a_posteriori(logits, classes=None):
    logits = np.atleast_2d(logits)
    predictions = np.argmax(logits, axis=1)
    if classes is not None:
        predictions = np.take(classes, predictions)
    return predictions

def init_esn_model(model_config, input_dim, audio_features, seed=None, workers=None, dtype=np.float64, **overrides):
    rpy.set_seed(seed)

    def get_p(key, default_attr, default_val):
        if key in overrides:
            return overrides[key]
        return getattr(model_config, default_attr, default_val)

    sr = get_p("sr", "sr", 0.4)
    leak = get_p("leak", "leak", 0.1)
    iss_val = get_p("iss", "iss", 0.0005)
    ridge = get_p("ridge", "ridge", 1e-8)
    isd_val = get_p("isd", "isd", 0.02)
    isd2_val = get_p("isd2", "isd2", 0.002)
    n_units = get_p("units", "units", 1000)

    # Construction du scaling block par block
    scalings = []
    if "mfcc" in audio_features:
        scalings.append(np.ones((input_dim,)) * iss_val)
    if "delta" in audio_features:
        scalings.append(np.ones((input_dim,)) * isd_val)
    if "delta2" in audio_features:
        scalings.append(np.ones((input_dim,)) * isd2_val)

    if not scalings:
        input_scaling = iss_val
    else:
        input_scaling = np.concatenate(scalings, axis=0)

    reservoir = Reservoir(
        n_units,
        sr=sr,
        lr=leak,
        input_scaling=input_scaling,
        bias=iss_val,
        W=fast_spectral_initialization,
        dtype=dtype,
    )

    readout = Ridge(ridge=ridge)
    n_workers = workers if workers is not None else getattr(model_config, "workers", -1)
    
    return ESN(
        reservoir=reservoir,
        readout=readout,
        workers=n_workers, 
        backend=getattr(model_config, "backend", "multiprocessing")
    )

def fit_esn_seq_by_seq(model, X_seqs, Y_seqs):
    """Memory-efficient ESN fit: interleave reservoir.run and readout.worker
    one sequence at a time so peak RAM = one state matrix instead of all of them.
    Mathematically equivalent to model.fit(X_seqs, Y_seqs).
    """
    model.fit([np.asarray(X_seqs[0])], [np.asarray(Y_seqs[0])])
    reservoir = model.reservoir
    readout   = model.readout

    def _gen():
        for x_seq, y_seq in zip(X_seqs, Y_seqs):
            reservoir.reset()
            states = np.asarray(reservoir.run(x_seq))
            yield readout.worker(states, y_seq)
            del states

    readout.master(_gen())


def predict_with_esn(annotator, corpus, return_raw=False, redo_transforms=False):
    if not hasattr(annotator, 'rpy_model'):
        raise NotTrainedError("Annotator does not contain a trained rpy_model.")

    corpus = annotator.transforms(
        corpus, purpose="annotation", output_directory=corpus.spec_directory, redo=redo_transforms,
    )
    notated_paths, mfccs = load_mfccs_for_annotation(corpus)
    raw_preds = annotator.rpy_model.run(mfccs)

    if isinstance(raw_preds, np.ndarray) and raw_preds.ndim < 3:
        raw_preds = [raw_preds]

    cls_preds = [maximum_a_posteriori(y, classes=annotator.vocab) for y in raw_preds]
    return notated_paths, cls_preds, raw_preds if return_raw else None