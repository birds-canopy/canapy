import gc
import logging
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
import traceback
import json
import threading

_IS_WINDOWS = sys.platform == "win32"
from typing import Dict
import numpy as np

from hyperopt import STATUS_OK, STATUS_FAIL

from reservoirpy.hyper import research, parallel_research

from canapy.corpus import Corpus
from canapy.annotator.commons.esn import init_esn_model
from canapy.timings import seconds_to_frames, seconds_to_audio

logger = logging.getLogger("canapy")

# =============================================================================
# 1. Helpers (kept for external use / backward compat)
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


def _arrays_to_memmap(arrays, tmpdir):
    """Write numpy arrays to memmap files and return read-only memmaps."""
    result = []
    for i, arr in enumerate(arrays):
        path = os.path.join(tmpdir, f"arr_{i}.dat")
        mm = np.memmap(path, dtype=arr.dtype, mode="w+", shape=arr.shape)
        mm[:] = arr
        mm.flush()
        result.append(np.memmap(path, dtype=arr.dtype, mode="r", shape=arr.shape))
    return result


# =============================================================================
# 2. Metadata extraction (no file I/O)
# =============================================================================

def _get_seq_metadata(corpus, purpose):
    """
    Extract per-sequence metadata from corpus.dataset without loading MFCC files.

    Returns
    -------
    seqids : list of str
    frame_estimates : dict  seqid -> int (upper-bound frame count from annotations)
    class_weights   : dict  seqid -> float (inverse-frequency weight for selection)
    df_prepared     : DataFrame with seqid / onset_spec / offset_spec columns added
    mfcc_paths      : DataFrame  (corpus.data_resources["syn_mfcc"])
    """
    split = "train" if purpose == "training" else "not train"
    df = corpus.dataset.query(split).copy()

    mfcc_paths = corpus.data_resources["syn_mfcc"]

    sampling_rate = corpus.config.transforms.audio.sampling_rate
    hop_length = seconds_to_audio(corpus.config.transforms.audio.hop_length, sampling_rate)

    df["onset_spec"] = seconds_to_frames(df["onset_s"], hop_length, sampling_rate)
    df["offset_spec"] = seconds_to_frames(df["offset_s"], hop_length, sampling_rate)
    df["seqid"] = df["sequence"].astype(str) + df["annotation"].astype(str)

    # Class-level frame counts for inverse-frequency weighting
    class_frame_counts = {}
    for label, group in df.groupby("label"):
        frames = int((group["offset_spec"] - group["onset_spec"]).sum())
        class_frame_counts[label] = max(1, frames)

    seqids = df["seqid"].unique().tolist()
    frame_estimates = {}
    class_weights = {}

    for seqid in seqids:
        seq_annots = df[df["seqid"] == seqid]
        frame_estimates[seqid] = int(seq_annots["offset_spec"].max())
        present = seq_annots["label"].unique()
        class_weights[seqid] = sum(1.0 / class_frame_counts[c] for c in present)

    return seqids, frame_estimates, class_weights, df, mfcc_paths


# =============================================================================
# 3. Frame-budget sequence selection
# =============================================================================

def _select_sequences_by_frames(seqids, frame_estimates, class_weights, max_percentage, rng):
    """
    Select sequences using inverse class-frequency weighting until a frame budget
    (max_percentage of total frames) is reached.

    Replaces the former count-based _select_representative_sequences so that
    max_percentage maps predictably to a fraction of the data volume in frames,
    not in sequence count.
    """
    if max_percentage >= 1.0:
        return list(seqids)

    total_frames = sum(frame_estimates[s] for s in seqids)
    max_frames = max(1, int(total_frames * max_percentage))

    weights = np.array([class_weights[s] for s in seqids], dtype=np.float64)
    total_w = weights.sum()
    if total_w == 0:
        weights = np.ones(len(seqids)) / len(seqids)
    else:
        weights /= total_w

    # Weighted shuffle without replacement; pick until frame budget is met
    order = rng.choice(len(seqids), size=len(seqids), replace=False, p=weights)

    selected = []
    accumulated = 0
    for idx in order:
        s = seqids[idx]
        selected.append(s)
        accumulated += frame_estimates[s]
        if accumulated >= max_frames:
            break

    logger.info(
        f"Frame-based selection: {len(selected)}/{len(seqids)} sequences, "
        f"~{accumulated}/{total_frames} frames "
        f"({accumulated / max(total_frames, 1) * 100:.1f}%)."
    )
    return selected


# =============================================================================
# 4. HP validation split (carved from training data)
# =============================================================================

def _split_hp_val(seqids, df_prepared, val_ratio, rng):
    """
    Split training seqids into hp_train and hp_val without touching the
    corpus train/test split.

    Guarantees that every class present in the training set is represented
    by at least one sequence in hp_train (so the ESN can always learn it).
    The remaining sequences are randomly assigned to respect val_ratio.
    """
    seqids = list(seqids)
    n = len(seqids)

    if n < 2:
        logger.warning("Too few training sequences to split for HP validation — using all for both.")
        return list(seqids), list(seqids)

    n_val_target = max(1, int(round(n * val_ratio)))
    n_train_target = n - n_val_target

    if n_train_target < 1:
        n_train_target = 1
        n_val_target = n - 1

    seqid_set = set(seqids)

    # For each class, pick one sequence that MUST stay in hp_train
    all_classes = df_prepared["label"].unique()
    must_train = set()
    for cls in all_classes:
        cls_seqids = [
            s for s in df_prepared[df_prepared["label"] == cls]["seqid"].unique()
            if s in seqid_set
        ]
        if not cls_seqids:
            continue
        chosen_idx = int(rng.integers(len(cls_seqids)))
        must_train.add(cls_seqids[chosen_idx])

    if len(must_train) >= n:
        logger.warning(
            "All training sequences are needed to cover every class — "
            "HP validation set mirrors HP training set."
        )
        return list(seqids), list(seqids)

    # Randomly shuffle the unconstrained sequences
    remaining = [s for s in seqids if s not in must_train]
    perm = rng.permutation(len(remaining))
    remaining = [remaining[int(i)] for i in perm]

    need_more = max(0, n_train_target - len(must_train))
    hp_train = list(must_train) + remaining[:need_more]
    hp_val = remaining[need_more:]

    # Safety: ensure hp_val is never empty
    if not hp_val:
        movable = [s for s in hp_train if s not in must_train]
        if movable:
            hp_val = [movable[0]]
            hp_train = [s for s in hp_train if s != movable[0]]
        else:
            hp_val = list(hp_train)

    logger.info(
        f"HP validation split (from train set): "
        f"{len(hp_train)} train seqs / {len(hp_val)} val seqs "
        f"(target ratio: {val_ratio:.0%}, actual: {len(hp_val)/n:.0%})."
    )
    return hp_train, hp_val


# =============================================================================
# 5. Direct memmap loader  (piste 1 + 4)
# =============================================================================

def _load_selected_seqids_to_memmap(
    corpus, selected_seqids, df_prepared, mfcc_paths, frame_estimates, n_classes, tmpdir, label
):
    """
    Load only the selected sequences and write them directly into pre-allocated
    memmaps — one MFCC file at a time, never accumulating lists.

    Peak RAM during this call:
        ~ max(one raw MFCC file) + the memmap backing file (disk).
    """
    if not selected_seqids:
        raise ValueError(f"No sequences selected for '{label}' loading.")

    # Infer n_features from the first file (single quick load)
    first_seqid = selected_seqids[0]
    first_seq = df_prepared[df_prepared["seqid"] == first_seqid]
    first_audio = first_seq["notated_path"].iloc[0]
    first_spec_path = (
        mfcc_paths.query("notated_path == @first_audio")["feature_path"].iloc[0]
    )
    tmp_mfcc = np.load(first_spec_path)
    n_features = tmp_mfcc["feature"].shape[0]
    del tmp_mfcc
    gc.collect()

    # Allocate memmaps with 5 % margin + small constant to absorb estimation error
    estimated_total = sum(frame_estimates[s] for s in selected_seqids)
    alloc_frames = int(estimated_total * 1.05) + 128

    x_path = os.path.join(tmpdir, f"X_{label}.dat")
    y_path = os.path.join(tmpdir, f"Y_{label}.dat")
    X_mm = np.memmap(x_path, dtype=np.float64, mode="w+", shape=(alloc_frames, n_features))
    Y_mm = np.memmap(y_path, dtype=np.float64, mode="w+", shape=(alloc_frames, n_classes))

    pos = 0
    seq_lengths = []  # actual frames written per sequence — used to rebuild list in objective
    for seqid in selected_seqids:
        seq_annots = df_prepared[df_prepared["seqid"] == seqid]
        notated_audio = seq_annots["notated_path"].iloc[0]
        notated_spec = (
            mfcc_paths.query("notated_path == @notated_audio")["feature_path"].iloc[0]
        )

        # Load one file, immediately free after writing to memmap
        raw = np.load(notated_spec)
        mfcc = raw["feature"].squeeze()  # (n_features, n_total_frames)
        del raw

        seq_end = min(int(seq_annots["offset_spec"].max()), mfcc.shape[1])

        mfcc_slice = mfcc[:, :seq_end].T.astype(np.float64)
        del mfcc

        repeated_labels = np.zeros((seq_end, n_classes), dtype=np.float64)
        for row in seq_annots.itertuples():
            onset = row.onset_spec
            offset = min(row.offset_spec, seq_end)
            repeated_labels[onset:offset] = row.encoded_label

        end = pos + seq_end
        if end > alloc_frames:
            # Safety clamp (should not happen with the 5 % margin)
            logger.warning(
                f"Memmap over-allocation for '{label}' at seqid {seqid}; truncating."
            )
            seq_end = alloc_frames - pos
            end = alloc_frames
            X_mm[pos:end] = mfcc_slice[:seq_end]
            Y_mm[pos:end] = repeated_labels[:seq_end]
        else:
            X_mm[pos:end] = mfcc_slice
            Y_mm[pos:end] = repeated_labels

        del mfcc_slice, repeated_labels
        gc.collect()

        seq_lengths.append(seq_end)  # track actual frames written for this sequence
        pos = end
        if pos >= alloc_frames:
            break

    X_mm.flush()
    Y_mm.flush()
    del X_mm, Y_mm

    # Open read-only views of the actually-used portion
    X_final = np.memmap(x_path, dtype=np.float64, mode="r", shape=(pos, n_features))
    Y_final = np.memmap(y_path, dtype=np.float64, mode="r", shape=(pos, n_classes))

    logger.info(
        f"Loaded '{label}': {pos} frames × {n_features} features "
        f"(float64 memmap, {pos * n_features * 8 / 1e6:.1f} MB on disk)."
    )
    return X_final, Y_final, seq_lengths


# =============================================================================
# 6. OBJECTIVE METHOD
# =============================================================================

def _fit_esn_seq_by_seq(model, X_seqs, Y_seqs):
    """
    Memory-efficient ESN fit: pipeline reservoir → Ridge one sequence at a time.

    model.fit(X_seqs, Y_seqs) first runs the reservoir on every sequence and
    keeps all state matrices in RAM simultaneously
    (n_total_frames × n_units × sizeof(float)), then feeds them to Ridge.
    With many long sequences and 8 parallel workers this saturates memory.

    Here we interleave reservoir.run and readout.worker: states for sequence N
    are computed, consumed by worker (→ tiny XtX / XtY), and deleted before
    sequence N+1 is processed.  Peak RAM per worker = one sequence's states
    instead of the whole dataset's states.
    """
    # One-sequence fit to trigger reservoirpy initialisation (sets input/output dims)
    model.fit([np.asarray(X_seqs[0])], [np.asarray(Y_seqs[0])])

    reservoir = model.reservoir
    readout   = model.readout

    def _gen():
        for x_seq, y_seq in zip(X_seqs, Y_seqs):
            reservoir.reset()
            states = np.asarray(reservoir.run(x_seq))
            yield readout.worker(states, y_seq)
            del states

    # Overwrite the Wout computed from the single-sequence init above
    readout.master(_gen())


def objective(dataset, config, **kwargs):
    """
    Method executed by each worker.
    - dataset : contains memmap file descriptors (path/shape/dtype), model
                config, and a sequential_esn flag (True when called from
                parallel_research, to prevent nested ESN multiprocessing).
                Arrays are NOT passed directly to avoid n_jobs-fold copies
                during loky/cloudpickle serialisation — workers re-open the
                memmap files from disk instead (zero-copy, OS page sharing).
    - config
    - kwargs : hyperparameters chosen for this run
    """
    (x_train_path, x_train_shape, x_train_dtype,
     y_train_path, y_train_shape, y_train_dtype,
     x_val_path,   x_val_shape,   x_val_dtype,
     y_val_path,   y_val_shape,   y_val_dtype,
     model_config, input_dim, audio_features, sequential_esn, *_rest) = dataset

    progress_file     = _rest[0] if len(_rest) > 0 else None
    train_seq_lengths = _rest[1] if len(_rest) > 1 else None
    val_seq_lengths   = _rest[2] if len(_rest) > 2 else None

    X_train = np.memmap(x_train_path, dtype=x_train_dtype, mode="r", shape=x_train_shape)
    Y_train = np.memmap(y_train_path, dtype=y_train_dtype, mode="r", shape=y_train_shape)
    X_val   = np.memmap(x_val_path,   dtype=x_val_dtype,   mode="r", shape=x_val_shape)
    Y_val   = np.memmap(y_val_path,   dtype=y_val_dtype,   mode="r", shape=y_val_shape)

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
            workers=1 if sequential_esn else None,
            dtype=np.float64,
            **kwargs
        )

        # Reconstruct per-sequence lists to match SynAnnotator.fit behavior:
        # passing a list lets ReservoirPy reset/isolate reservoir state between
        # sequences, identical to what happens during actual training.
        if train_seq_lengths is not None:
            pos, X_fit, Y_fit = 0, [], []
            for l in train_seq_lengths:
                X_fit.append(X_train[pos:pos + l])
                Y_fit.append(Y_train[pos:pos + l])
                pos += l
        else:
            X_fit, Y_fit = X_train, Y_train

        if isinstance(X_fit, list):
            _fit_esn_seq_by_seq(model, X_fit, Y_fit)
        else:
            model.fit(X_fit, Y_fit)
        predictions = model.predict(X_val)

        mse = np.mean((predictions - Y_val) ** 2)

        if np.isnan(mse) or np.isinf(mse):
            return {"loss": 1e5, "status": STATUS_FAIL}

        if progress_file:
            try:
                with open(progress_file, "ab") as _f:
                    _f.write(b"x")
            except Exception:
                pass

        return {"loss": float(mse), "status": STATUS_OK}

    except Exception as e:
        if progress_file:
            try:
                with open(progress_file, "ab") as _f:
                    _f.write(b"x")
            except Exception:
                pass
        return {"loss": float("inf"), "status": STATUS_FAIL, "error": str(e)}

# =============================================================================
# 7. MAIN PIPELINE
# =============================================================================

def optimize_hyperparameters(
    corpus: Corpus,
    config: Dict,
    annotator_type: str = "syn",
    n_iter: int = 100,
    max_percentage: float = 1.0,
    parallel: bool = False,
    n_jobs: int = 4,
    hp_val_ratio: float = 0.2,
    seed: int = 42,
    progress_file: str = None,
    progress_queue=None,
):
    logger.info("Starting hyperparameter optimization pipeline...")

    rng = np.random.default_rng(corpus.config.misc.seed)
    n_classes = len(corpus.dataset["label"].unique())

    # Always use a memmap directory (used both in parallel and sequential mode)
    mmap_tmpdir = tempfile.mkdtemp(prefix="canapy_mmap_")
    config_path = None

    try:
        # --- Step 1: extract metadata (no MFCC file I/O) ---
        logger.info("Step 1 (Opt): Extracting sequence metadata...")
        train_seqids, train_frame_est, train_class_w, train_df, mfcc_paths = \
            _get_seq_metadata(corpus, "training")

        if not train_seqids:
            logger.error("CRITICAL: No training sequences found!")
            return None

        # --- Step 2: carve a validation set out of the training data ---
        # The test set is never touched during HP search, consistent with the
        # fact that SynAnnotator.fit() trains on train data only (no val split).
        hp_train_seqids, hp_val_seqids = _split_hp_val(
            train_seqids, train_df, hp_val_ratio, rng
        )

        # --- Step 3: frame-based selection on hp_train only ---
        if max_percentage < 1.0:
            selected_train = _select_sequences_by_frames(
                hp_train_seqids, train_frame_est, train_class_w, max_percentage, rng
            )
        else:
            selected_train = list(hp_train_seqids)

        # --- Step 4: load directly to memmaps (piste 1 + 4) ---
        # One file at a time in RAM; no intermediate lists; float64 throughout.
        logger.info("Step 2 (Opt): Loading HP-train MFCCs to memmap...")
        X_train, Y_train, train_seq_lengths = _load_selected_seqids_to_memmap(
            corpus, selected_train, train_df, mfcc_paths,
            train_frame_est, n_classes, mmap_tmpdir, "train"
        )

        logger.info("Step 3 (Opt): Loading HP-val MFCCs to memmap...")
        X_val, Y_val, val_seq_lengths = _load_selected_seqids_to_memmap(
            corpus, hp_val_seqids, train_df, mfcc_paths,
            train_frame_est, n_classes, mmap_tmpdir, "val"
        )

        # Release any residual allocations before launching the search
        gc.collect()

        audio_features = corpus.config.transforms.audio.audio_features
        total_dim = X_train.shape[1]
        n_feats = len(audio_features)
        input_dim = total_dim // n_feats if n_feats > 0 else total_dim

        # Hyperparameters space definition
        hp_space = {
            "sr":   ["loguniform", 1e-3, 1e1],
            "leak": ["loguniform", 1e-3, 1.0],
            "ridge":["loguniform", 1e-9, 1e-2],
            "iss":  ["loguniform", 1e-4, 1e1],
            "isd":  ["loguniform", 1e-4, 1e1],
            "isd2": ["loguniform", 1e-4, 1e1],
            "seed": ["randint", 0, 10000],
        }

        research_config = {
            "exp": "canapy_opt",
            "hp_max_evals": n_iter,
            "hp_method": "random" if parallel else "tpe",
            "hp_space": hp_space,
            "instances_per_trial": 1,
            "seed": seed,
        }

        fd, config_path = tempfile.mkstemp(suffix=".json", prefix="canapy_opt_")
        os.close(fd)
        with open(config_path, "w") as f:
            json.dump(research_config, f)

        logger.info(f"Step 4 (Opt): Launching ReservoirPy research ({n_iter} iters)...")

        # Wrap objective to report per-trial progress (sequential mode only)
        if not parallel and progress_queue is not None:
            trial_count = [0]

            def _objective_tracked(dataset, config, **kwargs):
                result = objective(dataset, config, **kwargs)
                trial_count[0] += 1
                try:
                    progress_queue.put_nowait(trial_count[0])
                except Exception:
                    pass
                return result

            research_objective = _objective_tracked
        else:
            research_objective = objective

        # In parallel mode, use a file-based progress counter: each worker
        # appends one byte after completing a trial; file size == trials done.
        # A monitoring thread polls the file and forwards counts to progress_queue.
        progress_file_path = None
        _progress_stop = None
        _progress_monitor = None
        if parallel and progress_queue is not None:
            progress_file_path = os.path.join(mmap_tmpdir, "progress.bin")
            # Create the file so workers can open it immediately
            open(progress_file_path, "wb").close()

            _progress_stop = threading.Event()
            _last_count = [0]

            def _monitor_progress_file():
                while not _progress_stop.is_set():
                    try:
                        count = os.path.getsize(progress_file_path)
                        if count > _last_count[0]:
                            _last_count[0] = count
                            try:
                                progress_queue.put_nowait(count)
                            except Exception:
                                pass
                    except OSError:
                        pass
                    time.sleep(0.5)

            _progress_monitor = threading.Thread(target=_monitor_progress_file, daemon=True)
            _progress_monitor.start()

        # Pass only file paths + shapes + dtypes, never the arrays themselves.
        # loky/cloudpickle would serialize numpy.memmap as full array data,
        # causing n_jobs copies in RAM at worker startup. Each worker instead
        # re-opens a read-only memmap handle pointing to the same backing file;
        # the OS shares pages across all workers at no extra memory cost.
        dataset_tuple = (
            X_train.filename, X_train.shape, X_train.dtype,
            Y_train.filename, Y_train.shape, Y_train.dtype,
            X_val.filename,   X_val.shape,   X_val.dtype,
            Y_val.filename,   Y_val.shape,   Y_val.dtype,
            config.model.syn, input_dim, audio_features,
            parallel,  # sequential_esn=True only when inside parallel workers
            progress_file_path,
            train_seq_lengths,
            val_seq_lengths,
        )

        try:
            if parallel:
                if hasattr(threading, "_SHUTTING_DOWN"):
                    threading._SHUTTING_DOWN = False
                best_params, _trials = parallel_research(
                    research_objective,
                    dataset_tuple,
                    config_path,
                    n_jobs=n_jobs,
                )
            else:
                best_params, _trials = research(
                    research_objective,
                    dataset_tuple,
                    config_path,
                )
        finally:
            if _progress_stop is not None:
                _progress_stop.set()
            if _progress_monitor is not None:
                _progress_monitor.join(timeout=2)

        return best_params

    except Exception as e:
        logger.error(f"Global Optimization Crash: {str(e)}")
        traceback.print_exc()
        return None

    finally:
        if mmap_tmpdir and os.path.exists(mmap_tmpdir):
            shutil.rmtree(mmap_tmpdir, ignore_errors=True)
        if config_path and os.path.exists(config_path):
            try:
                os.remove(config_path)
            except OSError:
                pass


# =============================================================================
# 8. ISOLATED SUBPROCESS WRAPPER
# =============================================================================

def _subprocess_target(queue, corpus, config, annotator_type, n_iter, max_percentage, parallel, n_jobs, hp_val_ratio=0.2, seed=42, progress_queue=None):
    """
    Entry point for the isolated optimization subprocess.
    Must be a module-level function so multiprocessing can pickle it.

    On Unix: calls os.setpgrp() to create a new process group whose ID equals
    this process's PID. Every child spawned from here (loky workers, etc.)
    inherits that PGID. The parent can therefore kill the entire group with a
    single os.killpg() call.
    On Windows: process groups are not supported; the parent tracks the tree
    via psutil instead.
    """
    # Own process group — all descendants inherit this PGID (Unix only).
    if not _IS_WINDOWS:
        os.setpgrp()

    # Lower priority so the UI stays responsive during a long search.
    try:
        if not _IS_WINDOWS:
            os.nice(10)
    except OSError:
        pass

    # Cap BLAS/OpenMP threads per worker to prevent oversubscription.
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
            hp_val_ratio=hp_val_ratio,
            seed=seed,
            progress_queue=progress_queue,
        )
        queue.put(("ok", result))
    except MemoryError:
        msg = (
            "Dataset too large for optimization on your machine. "
            "Try reducing the dataset percentage or the number of parallel jobs."
        )
        queue.put(("error", msg, traceback.format_exc()))
    except Exception as e:
        queue.put(("error", str(e), traceback.format_exc()))
    finally:
        # Bypass Python atexit handlers (notably joblib/loky worker pool teardown)
        # which would block for minutes waiting for workers to exit gracefully.
        # The parent process kills all survivors via _killpg_safe(pgid) anyway.
        queue.close()
        queue.join_thread()  # ensure pipe is flushed before exiting
        os._exit(0)


def _killpg_safe(pgid):
    """Kill all processes in the group (Unix) or tree (Windows), ignoring errors if already gone."""
    if _IS_WINDOWS:
        import psutil
        try:
            root = psutil.Process(pgid)
            children = root.children(recursive=True)
            for child in children:
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            try:
                root.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    else:
        import signal
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass  # process group already gone — clean exit


def _rss_of_group(pgid):
    """Return total RSS (bytes) of the subprocess and all its descendants.

    On Unix: sums all processes sharing the process group ID (pgid == subprocess PID).
    On Windows: walks the process tree rooted at pgid (== subprocess PID) via psutil.
    """
    import psutil
    total = 0
    if _IS_WINDOWS:
        try:
            root = psutil.Process(pgid)
            for proc in [root] + root.children(recursive=True):
                try:
                    total += proc.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    else:
        for proc in psutil.process_iter(["pid", "memory_info"]):
            try:
                if os.getpgid(proc.pid) == pgid:
                    total += proc.memory_info().rss
            except (OSError, psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    return total


def optimize_hyperparameters_isolated(
    corpus: Corpus,
    config: Dict,
    annotator_type: str = "syn",
    n_iter: int = 100,
    max_percentage: float = 1.0,
    parallel: bool = False,
    n_jobs: int = 4,
    hp_val_ratio: float = 0.2,
    seed: int = 42,
    progress_callback=None,
    timeout : int = None,
):
    """
    Run optimize_hyperparameters in a fresh spawned subprocess.

    The subprocess calls os.setpgrp() immediately, putting itself and all its
    descendants (loky workers, etc.) in a dedicated process group whose ID
    equals the subprocess PID.  This guarantees that a single os.killpg() call
    wipes every process in the group — even if the subprocess has already
    exited and workers were reparented to init.

    Cleanup flow (always runs in the finally block):
      1. Stop helper threads (watchdog, progress poller).
      2. os.killpg(pgid, SIGKILL) — kills any survivors atomically.
      3. p.join() — reaps the subprocess zombie entry.

    progress_callback(done: int, total: int) is called from a polling thread
    after each completed trial (sequential mode only).
    """
    import psutil
    import time as _time

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()

    # Leave at least 2 GB free for the OS + UI.
    _SAFETY_HEADROOM = 2 * 1024 ** 3
    rss_threshold = max(
        512 * 1024 ** 2,
        psutil.virtual_memory().available - _SAFETY_HEADROOM,
    )
    logger.info(
        f"RSS watchdog threshold: {rss_threshold / 1e9:.1f} GB "
        f"(available: {psutil.virtual_memory().available / 1e9:.1f} GB)"
    )

    progress_queue = ctx.Queue() if progress_callback is not None else None

    p = ctx.Process(
        target=_subprocess_target,
        args=(queue, corpus, config, annotator_type, n_iter, max_percentage, parallel, n_jobs, hp_val_ratio, seed, progress_queue),
        daemon=False,
    )
    p.start()
    pgid = p.pid  # subprocess calls setpgrp() so its PGID == its PID

    stop_event = threading.Event()
    oom_terminated = threading.Event()

    def _watchdog():
        while not stop_event.is_set():
            if _rss_of_group(pgid) > rss_threshold:
                logger.error(
                    f"Watchdog: process group {pgid} exceeded RSS threshold "
                    f"{rss_threshold / 1e9:.1f} GB — sending SIGKILL to group."
                )
                oom_terminated.set()
                _killpg_safe(pgid)
                return
            _time.sleep(0.5)

    def _poll():
        while not stop_event.is_set():
            try:
                while not progress_queue.empty():
                    done = progress_queue.get_nowait()
                    try:
                        progress_callback(done, n_iter)
                    except Exception:
                        pass
            except Exception:
                pass
            _time.sleep(0.3)

    watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
    watchdog_thread.start()
    poll_thread = None
    if progress_queue is not None:
        poll_thread = threading.Thread(target=_poll, daemon=True)
        poll_thread.start()

    try:
        p.join(timeout=timeout)
    finally:
        # Always executed: stop threads, then nuke the entire process group.
        # This catches every case: normal exit, watchdog kill, timeout, exception.
        stop_event.set()
        watchdog_thread.join(timeout=2)
        if poll_thread is not None:
            poll_thread.join(timeout=2)

        # SIGKILL the whole group — no-op if everyone already exited cleanly.
        _killpg_safe(pgid)

        # Reap the subprocess if still technically alive after the kill.
        if p.is_alive():
            p.join(timeout=5)

    if oom_terminated.is_set():
        raise RuntimeError(
            "Dataset too large for optimization on your machine. "
            "Try reducing the dataset percentage or the number of parallel jobs."
        )

    if timeout is not None and not p.is_alive() and p.exitcode is not None and p.exitcode < 0:
        # Killed by signal (e.g. timeout SIGKILL) without putting anything on queue
        logger.error(f"Optimization subprocess killed by signal {-p.exitcode}.")
        return None

    if not queue.empty():
        status, *data = queue.get_nowait()
        if status == "ok":
            return data[0]
        else:
            logger.error(f"Optimization subprocess failed: {data[0]}")
            if len(data) > 1:
                logger.error(data[1])
            raise RuntimeError(data[0])
    return None
