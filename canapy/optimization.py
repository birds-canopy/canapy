import logging
import os
import tempfile
import traceback
import json
import warnings
import time
from glob import glob
from typing import Optional, Dict
import numpy as np

from joblib import Parallel, cpu_count, delayed, parallel_config
from tqdm import tqdm

from hyperopt import STATUS_OK, STATUS_FAIL

from canapy.corpus import Corpus
from canapy.annotator.commons.esn import init_esn_model
from canapy.annotator.commons.mfccs import load_mfccs_and_repeat_labels
from canapy.transforms.commons.training import encode_labels

logger = logging.getLogger("canapy")

# =============================================================================
# 1. Parsing utils
# =============================================================================

def rand_generator(seed=None):
    return np.random.default_rng(seed)

def _parse_searchspace(specs):
    """Converts JSON lists in executables methods"""
    if not isinstance(specs, list):
        return lambda rng: specs

    dist = specs[0]
    
    # Note: Theses lambdas are only used in the principal process 
    # to generate params list. Wont get sent to workers. 
    if dist == "choice":
        choices = specs[1:]
        return lambda rng: choices[rng.integers(0, len(choices))]
        
    if dist == "randint":
        return lambda rng: rng.integers(specs[1], specs[2])
    if dist == "uniform":
        return lambda rng: rng.uniform(specs[1], specs[2])
    if dist == "quniform":
        return lambda rng: np.round(rng.uniform(specs[1], specs[2]) / specs[3]) * specs[3]
    if dist == "loguniform":
        return lambda rng: np.exp(rng.uniform(np.log(specs[1]), np.log(specs[2])))
    if dist == "qloguniform":
        return lambda rng: np.round(np.exp(rng.uniform(np.log(specs[1]), np.log(specs[2]))) / specs[3]) * specs[3]
    if dist == "normal":
        return lambda rng: specs[1] + specs[2] * rng.randn()
    
    return lambda rng: specs

def _parse_config(config):
    """Validate and transform HP space."""
    if "hp_space" in config:
        config["hp_space"] = {arg: _parse_searchspace(specs) for arg, specs in config["hp_space"].items()}
    return config

def _get_conf_from_json(confpath):
    with open(confpath, "r") as f:
        config = json.load(f)
    return _parse_config(config)

# =============================================================================
# 2. Backend Multiprocessing
# =============================================================================

def _worker(objective, dataset, config, kwargs):
    """Worker function executing the objective for one trial."""
    start = time.time()
    try:
        returned_dict = objective(dataset, config, **kwargs)
        status = returned_dict.get("status", STATUS_OK)
    except Exception as e:
        traceback.print_exc()
        returned_dict = {"loss": float("inf"), "status": STATUS_FAIL, "error": str(e)}
        status = STATUS_FAIL
    
    end = time.time()
    returned_dict["start_time"] = start
    returned_dict["duration"] = end - start
    
    loss_val = returned_dict.get("loss", 1e9)
    save_file = f"{loss_val:.7f}_results" if status == STATUS_OK else f"ERR{start}_results"
        
    return kwargs, returned_dict, save_file

def custom_parallel_research(
    objective,
    dataset,
    config_path,
    report_path=None,
    n_jobs=-1,
    inner_max_num_threads=None, 
):
    """
    Parallel research from reservoirpy but customed to work inside panel
    """
    # 1. config parsing
    config = _get_conf_from_json(config_path)
    
    if report_path is None:
        report_path = os.path.join(tempfile.gettempdir(), config.get("exp", "exp"))
    os.makedirs(report_path, exist_ok=True)

    search_space = config["hp_space"]
    max_evals = config["hp_max_evals"]
    rng = rand_generator(config.get("seed"))

    # 2. genrating all parameters in the principal process
    # params_list contains int/float (pickable)
    params_list = [{arg: f(rng) for arg, f in search_space.items()} for _ in range(max_evals)]

    best_params = None
    best_loss = float("inf")

    logger.info(f"Parallel Research: launching {max_evals} tasks (Backend: Multiprocessing).")
    logger.info("NOTE: Progress bar will update only at the END of all tasks due to backend limitations.")

    with parallel_config(backend="multiprocessing"):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_worker)(objective, dataset, {}, kwargs) for kwargs in params_list
        )

        # 3. Handling results once everything is done
        pbar = tqdm(iterable=results, total=len(params_list), unit="trial", postfix="best loss=?")

        for kwargs, returned_dict, save_file_base in pbar:
            try:
                # Normalisation of numpy types for JSON
                for key in kwargs:
                    if isinstance(kwargs[key], (np.integer, np.int64, np.int32)):
                        kwargs[key] = int(kwargs[key])
                    elif isinstance(kwargs[key], (np.floating, np.float64, np.float32)):
                        kwargs[key] = float(kwargs[key])

                json_dict = {"returned_dict": returned_dict, "current_params": kwargs}
                full_save_path = os.path.join(report_path, save_file_base)
                
                nb_same = len(glob(f"{full_save_path}*"))
                final_name = f"{full_save_path}_{nb_same+1}call.json"
                
                with open(final_name, "w+") as f:
                    json.dump(json_dict, f, indent=2)
            except Exception as e:
                warnings.warn(f"JSON save failed: {e}")

            current_loss = returned_dict.get("loss", float("inf"))
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = kwargs
                pbar.set_postfix({"best loss": f"{best_loss:.5f}"})

    return best_params, best_loss

# =============================================================================
# 3. OBJECTIVE METHOD
# =============================================================================

def objective(dataset, config, **kwargs):
    """
    Method executed by each worker. 
    - dataset : contains data and model config
    - config 
    - kwargs : hyperparameters chosen for this run
    """
    X_train, Y_train, model_config, input_dim, audio_features = dataset

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


        split = int(len(X_train) * 0.8)
        X_t, X_v = X_train[:split], X_train[split:]
        Y_t, Y_v = Y_train[:split], Y_train[split:]
        
        # Train and validate
        model.fit(X_t, Y_t)
        predictions = model.predict(X_v)
        
        mse = np.mean((predictions - Y_v) ** 2)
            
        if np.isnan(mse) or np.isinf(mse):
            return {"loss": 1e5, "status": STATUS_FAIL}
            
        return {"loss": float(mse), "status": STATUS_OK}

    except Exception as e:
        return {"loss": float("inf"), "status": STATUS_FAIL, "error": str(e)}

# =============================================================================
# 4. MAIN PIPELINE
# =============================================================================

def optimize_hyperparameters(
    corpus: Corpus,
    config: Dict,
    annotator_type: str = "syn",
    n_iter: int = 100,
    max_samples: Optional[int] = None,
):
    logger.info("Starting hyperparameter optimization pipeline...")

    # Prepare data
    if "encoded_labels" not in corpus.data_resources:
        corpus = encode_labels(corpus, resource_name="encoded_labels")

    if "syn_mfcc" in corpus.data_resources:
        corpus.data_resources["mfcc"] = corpus.data_resources["syn_mfcc"]

    logger.info("Step 2 (Opt): Loading MFCCs...")
    _, _, X_list, Y_list = load_mfccs_and_repeat_labels(corpus, purpose="training")

    if not X_list or len(X_list) == 0:
        logger.error("CRITICAL: No training data found!")
        return None

    
    X_concat, Y_concat = [], []
    total_frames = 0
    indices = np.random.permutation(len(X_list))

    for i in indices:
        x, y = X_list[i], Y_list[i]
        min_len = min(x.shape[0], y.shape[0])
        
        y_cut = y[:min_len]
        if y_cut.ndim == 1:
            y_cut = y_cut.reshape(-1, 1)

        X_concat.append(x[:min_len])
        Y_concat.append(y_cut)

        total_frames += min_len
        if max_samples and total_frames >= max_samples:
            break

    if not X_concat:
        logger.error("Failed to concatenate data.")
        return None

    X_train = np.concatenate(X_concat)
    Y_train = np.concatenate(Y_concat)

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
            "isd2": ["uniform", 1e-3, 1e1],
            "seed": ["choice", 42]
        }

    research_config = {
        "exp": "canapy_opt",
        "hp_max_evals": n_iter,
        "hp_method": "random", 
        "hp_space": hp_space,
        "instances_per_trial": 3, 
        "seed": 42
    }

    # temporary JSON file for research config
    fd, config_path = tempfile.mkstemp(suffix=".json", prefix="canapy_opt_")
    os.close(fd) 

    with open(config_path, "w") as f:
        json.dump(research_config, f)

    try:
        logger.info(f"Step 3 (Opt): Launching Custom Parallel Research ({n_iter} iters)...")

        dataset_tuple = (X_train, Y_train, config.model.syn, input_dim, audio_features)

        
        best_params, best_loss = custom_parallel_research(
            objective,
            dataset_tuple,
            config_path,
            n_jobs=-1 
        )
        
        if best_params:
            best_params["seed"] = 42

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