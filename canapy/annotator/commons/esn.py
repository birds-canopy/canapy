# Author: Nathan Trouvain at 04/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: BSD-3-Clause
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
    """Memory-efficient, sequence-by-sequence offline training of an ESN readout.

    Mathematically equivalent to ``model.fit(X_seqs, Y_seqs)`` but with a much
    smaller peak memory footprint.

    Why 
    ---------------
    Training the Ridge readout solves the Tikhonov regression

        Wout = (XᵀX + λI)⁻¹ XᵀY

    where X is the stacked matrix of reservoir states over all training
    timesteps (shape [total_timesteps, n_units]) and Y the matching
    targets. The solution only depends on two fixed-size "sufficient
    statistics" that are independent of the number of timesteps:

        - XᵀX  of shape [n_units, n_units]    
        - XᵀY  of shape [n_units, n_outputs]

    ReservoirPy's Ridge.worker/Ridge.master already build Wout from
    these per-sequence partial sums (worker reduces one sequence to its
    XᵀX/XᵀY contribution, master accumulates them and solves). The
    problem is the driver: Model.fit runs the reservoir over the whole
    dataset and stores the entire state matrix X in memory before handing
    it to the readout. For long songs / large corpora, that stacked X is the
    memory bottleneck : its size grows with the total number of frames.

    What this does instead
    -------------------------------
    It keeps the same math but never materialises the full X: it processes
    one sequence at a time, run the reservoir, immediately reduce the resulting
    states to their XᵀX/XᵀY contribution, then free them. Peak RAM is
    therefore the state matrix of a single sequence, not of the whole dataset.

    See "A practical guide to applying Echo State Networks" by Lukosevicius for ref.
    
    Parameters
    ----------
    model : reservoirpy.ESN
        An ESN (reservoir + Ridge readout). Modified in place; its readout is
        trained when the call returns.
    X_seqs, Y_seqs : sequence of arrays
        One input array and one target array per training sequence, each of
        shape [timesteps, features] / [timesteps, n_outputs].
    """
    # Run a throwaway fit on a single sequence to infer readout input/output dimensions. 
    # Overwritten by readout.master() call
    model.fit([np.asarray(X_seqs[0])], [np.asarray(Y_seqs[0])])
    reservoir = model.reservoir
    readout   = model.readout

    def _gen():
        # Lazily yield each sequence's partial sufficient statistics. Because
        # this is a generator, only one sequence's states live in memory at a
        # time: master() pulls them one by one and accumulates immediately.
        for x_seq, y_seq in zip(X_seqs, Y_seqs):
            # Clear the reservoir's internal state so each sequence is encoded
            # independently, starting from a zero state (matching how Model.fit
            # treats a list of sequences as independent series).
            reservoir.reset()
            # Encode this one sequence: states has shape [timesteps, n_units].
            # This is the only large array kept alive at any moment.
            states = np.asarray(reservoir.run(x_seq))
            # Reduce the sequence to its fixed-size contribution to the normal
            # equations: worker() returns (XᵀX, XᵀY, sum_x, sum_y, n_samples)
            # for this sequence. Sizes no longer depend on timesteps.
            yield readout.worker(states, y_seq)
            # Drop the big state matrix right away so its memory can be reused
            # by the next sequence. this is what bounds peak RAM.
            del states

    # master() consumes the generator, summing every sequence's XᵀX and XᵀY into
    # the global accumulators, applies the ridge term and (optional) bias
    # centering, then solves for Wout. The result is identical to the one
    # model.fit() would have produced from the full stacked X.
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