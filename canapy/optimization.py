import logging
import os
import tempfile
import traceback
import json
from typing import Dict
import numpy as np

from hyperopt import STATUS_OK, STATUS_FAIL

from reservoirpy.hyper import research

from canapy.corpus import Corpus
from canapy.annotator.commons.esn import init_esn_model
from canapy.annotator.commons.mfccs import load_mfccs_and_repeat_labels
from canapy.transforms.synesn import SynESNTransform

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
# 2. OBJECTIVE METHOD
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
):
    logger.info("Starting hyperparameter optimization pipeline...")

    # Step 1: Apply the same transforms as SynAnnotator.fit()
    # This ensures: annotation sorting/merging/silence tagging,
    # train/test split (split_train_test), label encoding, and MFCC computation.
    logger.info("Step 1 (Opt): Applying SynESNTransform (same as real training)...")
    syn_transform = SynESNTransform()
    corpus = syn_transform(
        corpus,
        purpose="training",
        output_directory=corpus.spec_directory,
    )

    # Step 2: Load train and val data from the corpus split (same as SynAnnotator)
    logger.info("Step 2 (Opt): Loading training and validation MFCCs...")
    _, _, X_train_list, Y_train_list = load_mfccs_and_repeat_labels(corpus, purpose="training")
    _, _, X_val_list, Y_val_list = load_mfccs_and_repeat_labels(corpus, purpose="eval")

    if not X_train_list:
        logger.error("CRITICAL: No training data found!")
        return None
    if not X_val_list:
        logger.error("CRITICAL: No validation data found!")
        return None

    # Step 3: Select a subset of training sequences according to max_percentage
    if max_percentage < 1.0:
        rng = np.random.default_rng(corpus.config.misc.seed)
        n_total = len(X_train_list)
        n_select = max(1, int(n_total * max_percentage))
        indices = rng.choice(n_total, size=n_select, replace=False)
        X_train_list = [X_train_list[i] for i in indices]
        Y_train_list = [Y_train_list[i] for i in indices]
        logger.info(
            f"Using {n_select}/{n_total} training sequences ({max_percentage * 100:.0f}%)."
        )

    # Step 4: Concatenate sequences into single arrays
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
        "hp_method": "random",
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
        logger.info(f"Step 3 (Opt): Launching ReservoirPy research ({n_iter} iters)...")

        dataset_tuple = (X_train, Y_train, X_val, Y_val, config.model.syn, input_dim, audio_features)

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
