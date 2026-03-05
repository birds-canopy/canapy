import logging
import multiprocessing
import os
import tempfile
import traceback
import json
import threading
from typing import Dict
import numpy as np

from hyperopt import STATUS_OK, STATUS_FAIL

from reservoirpy.hyper import research, parallel_research

from canapy.corpus import Corpus
from canapy.annotator.commons.esn import init_esn_model
from canapy.annotator.commons.mfccs import load_mfccs_and_repeat_labels

logger = logging.getLogger("canapy")

# =============================================================================
# 1. Helper
# =============================================================================

def _concat_xy(X_list, Y_list):
    """Concatenate lists of sequence arrays, trimming to min length per sequence."""
    xs, ys = [], []
    for x, y in zip(X_list, Y_list):
        n = min(x.shape[0], y.shape[0])
        y_row = y[:n]
        if y_row.ndim == 1:
            y_row = y_row.reshape(-1, 1)
        xs.append(x[:n])
        ys.append(y_row)
    return np.concatenate(xs), np.concatenate(ys)

# =============================================================================
# 2. STRATIFIED SEQUENCE SELECTION
# =============================================================================

def _select_representative_sequences(X_list, Y_list, n_select, rng):
    """
    Select n_select sequences using inverse class-frequency weighting.

    Each sequence is assigned a score = sum(1/global_freq[c]) for every class c
    present in it. Sequences carrying rare classes thus get a higher sampling
    probability, ensuring the selected subset reflects the full class distribution
    better than uniform random sampling.

    Y arrays are expected to have shape (n_timesteps, n_classes) with one-hot
    encoded labels, as produced by load_mfccs_and_repeat_labels.
    """
    n_total = len(X_list)
    if n_select >= n_total:
        return X_list, Y_list

    # Global frame counts per class across the full training set
    n_classes = Y_list[0].shape[1]
    global_counts = np.zeros(n_classes)
    for y in Y_list:
        global_counts += y.sum(axis=0)

    # Avoid division by zero for classes absent from training (shouldn't happen)
    global_counts = np.maximum(global_counts, 1.0)

    # Per-sequence score: sum of inverse frequencies of classes present
    seq_weights = np.zeros(n_total)
    for i, y in enumerate(Y_list):
        present = y.sum(axis=0) > 0
        if present.any():
            seq_weights[i] = np.sum(1.0 / global_counts[present])

    total = seq_weights.sum()
    if total == 0:
        # Fallback to uniform if all weights collapse (degenerate case)
        seq_weights = np.ones(n_total)
        total = float(n_total)
    seq_weights /= total

    indices = rng.choice(n_total, size=n_select, replace=False, p=seq_weights)
    return [X_list[i] for i in indices], [Y_list[i] for i in indices]


# =============================================================================
# 3. OBJECTIVE METHOD
# =============================================================================

def objective(dataset, config, **kwargs):
    """
    Method executed by each worker.
    - dataset : contains train data, val data and model config
    - config
    - kwargs : hyperparameters chosen for this run
    """
    X_train, Y_train, X_val, Y_val, model_config, input_dim, audio_features = dataset

    try:
        if "seed" in kwargs:
             if isinstance(kwargs["seed"], (list, tuple)):
                 kwargs["seed"] = int(kwargs["seed"][0])
             else:
                 kwargs["seed"] = int(kwargs["seed"])
        else:
             kwargs["seed"] = 42

        model = init_esn_model(
            model_config,
            input_dim,
            audio_features,
            **kwargs
        )

        # Use the corpus train/test split — no internal split
        model.fit(X_train, Y_train)
        predictions = model.predict(X_val)

        mse = np.mean((predictions - Y_val) ** 2)

        if np.isnan(mse) or np.isinf(mse):
            return {"loss": 1e5, "status": STATUS_FAIL}

        return {"loss": float(mse), "status": STATUS_OK}

    except Exception as e:
        return {"loss": float("inf"), "status": STATUS_FAIL, "error": str(e)}

# =============================================================================
# 3. MAIN PIPELINE
# =============================================================================

def optimize_hyperparameters(
    corpus: Corpus,
    config: Dict,
    annotator_type: str = "syn",
    n_iter: int = 100,
    max_percentage: float = 1.0,
    parallel: bool = False,
    n_jobs: int = 4,
):
    logger.info("Starting hyperparameter optimization pipeline...")

    # Load train and val data from the corpus split.
    logger.info("Step 1 (Opt): Loading training and validation MFCCs...")
    _, _, X_train_list, Y_train_list = load_mfccs_and_repeat_labels(corpus, purpose="training")
    _, _, X_val_list, Y_val_list = load_mfccs_and_repeat_labels(corpus, purpose="eval")

    if not X_train_list:
        logger.error("CRITICAL: No training data found!")
        return None
    if not X_val_list:
        logger.error("CRITICAL: No validation data found!")
        return None

    # Select a representative subset of training sequences
    if max_percentage < 1.0:
        rng = np.random.default_rng(corpus.config.misc.seed)
        n_total = len(X_train_list)
        n_select = max(1, int(n_total * max_percentage))
        X_train_list, Y_train_list = _select_representative_sequences(
            X_train_list, Y_train_list, n_select, rng
        )
        logger.info(
            f"Selected {n_select}/{n_total} representative sequences "
            f"({max_percentage * 100:.0f}%, inverse-frequency weighted)."
        )

    # Concatenate sequences into single arrays
    X_train, Y_train = _concat_xy(X_train_list, Y_train_list)
    X_val, Y_val = _concat_xy(X_val_list, Y_val_list)

    audio_features = corpus.config.transforms.audio.audio_features
    total_dim = X_train.shape[1]
    n_feats = len(audio_features)
    input_dim = total_dim // n_feats if n_feats > 0 else total_dim

    # Hyperparameters space definition
    hp_space= {
            "sr": ["loguniform", 1e-3, 1e1],
            "leak": ["loguniform", 1e-3, 1.0],
            "ridge": ["loguniform", 1e-9, 1e-2],
            "iss": ["loguniform", 1e-3, 1e1],
            "isd": ["loguniform", 1e-3, 1e1],
            "isd2": ["loguniform", 1e-3, 1e1],
            "seed": ["randint", 0, 10000]
        }

    research_config = {
        "exp": "canapy_opt",
        "hp_max_evals": n_iter,
        "hp_method": "random" if parallel else "tpe",
        "hp_space": hp_space,
        "instances_per_trial": 1,
        "seed": 42
    }

    # temporary JSON file for research config
    fd, config_path = tempfile.mkstemp(suffix=".json", prefix="canapy_opt_")
    os.close(fd)

    with open(config_path, "w") as f:
        json.dump(research_config, f)

    try:
        logger.info(f"Step 2 (Opt): Launching ReservoirPy research ({n_iter} iters)...")

        dataset_tuple = (X_train, Y_train, X_val, Y_val, config.model.syn, input_dim, audio_features)
        if parallel:
            # Panel/Tornado callbacks can leave threading._SHUTTING_DOWN=True,
            # which causes loky/concurrent.futures to refuse creating new executors.
            # Reset it here since the process is not actually shutting down.
            if hasattr(threading, "_SHUTTING_DOWN"):
                threading._SHUTTING_DOWN = False
            best_params, _trials = parallel_research(
                objective,
                dataset_tuple,
                config_path,
                n_jobs=n_jobs,
            )
        else:
            best_params, _trials = research(
                objective,
                dataset_tuple,
                config_path,
            )

        return best_params

    except Exception as e:
        logger.error(f"Global Optimization Crash: {str(e)}")
        traceback.print_exc()
        return None

    finally:
        if config_path and os.path.exists(config_path):
            try:
                os.remove(config_path)
            except OSError:
                pass


# =============================================================================
# 4. ISOLATED SUBPROCESS WRAPPER
# =============================================================================

def _subprocess_target(queue, corpus, config, annotator_type, n_iter, max_percentage, parallel, n_jobs):
    """
    Entry point for the isolated optimization subprocess.
    Must be a module-level function so multiprocessing can pickle it.
    """
    # Lower priority so the UI stays responsive during a long search.
    try:
        os.nice(10)
    except OSError:
        pass

    # Cap BLAS/OpenMP threads per worker to prevent oversubscription.
    # Without this, each of the n_jobs workers would spawn cpu_count threads
    # (via OpenBLAS/MKL), saturating the machine (e.g. 8 workers × 16 threads = 128).
    if parallel and n_jobs > 1:
        threads_per_worker = max(1, (os.cpu_count() or 1) // n_jobs)
    else:
        threads_per_worker = os.cpu_count() or 1
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads_per_worker)

    try:
        result = optimize_hyperparameters(
            corpus, config,
            annotator_type=annotator_type,
            n_iter=n_iter,
            max_percentage=max_percentage,
            parallel=parallel,
            n_jobs=n_jobs,
        )
        queue.put(("ok", result))
    except Exception as e:
        queue.put(("error", str(e), traceback.format_exc()))


def optimize_hyperparameters_isolated(
    corpus: Corpus,
    config: Dict,
    annotator_type: str = "syn",
    n_iter: int = 100,
    max_percentage: float = 1.0,
    parallel: bool = False,
    n_jobs: int = 4,
):
    """
    Run optimize_hyperparameters in a fresh spawned subprocess.

    When parallel_research forks workers, it does so from a clean ~50 MB
    Python process instead of forking directly from VS Code's heavy process
    (Panel server + extensions...).  This prevents OOM crashes regardless of
    dataset size or n_jobs value.
    """
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(
        target=_subprocess_target,
        args=(queue, corpus, config, annotator_type, n_iter, max_percentage, parallel, n_jobs),
        daemon=False,  # must be False: daemon processes cannot spawn children
    )
    _TIMEOUT_S = 7200  # 2-hour safety net, prevents infinite hang on subprocess deadlock
    p.start()
    p.join(timeout=_TIMEOUT_S)

    if p.is_alive():
        logger.error(
            f"Optimization subprocess exceeded timeout ({_TIMEOUT_S}s). Terminating."
        )
        p.terminate()
        p.join(timeout=10)
        if p.is_alive():
            p.kill()
        return None

    if not queue.empty():
        status, *data = queue.get_nowait()
        if status == "ok":
            return data[0]
        else:
            logger.error(f"Optimization subprocess failed: {data[0]}")
            if len(data) > 1:
                logger.error(data[1])
    return None
