import tempfile
import uuid
import shutil
import warnings
from pathlib import Path

import numpy as np
from joblib import dump, load


TEMP_FOLDER = tempfile.gettempdir() + "/canapy"

if not Path(TEMP_FOLDER).exists():
    try:
        Path(TEMP_FOLDER).mkdir(parents=True)
    except OSError as e:
        print(f"Unable to create temp directory: {e}")
        TEMP_FOLDER = tempfile.gettempdir()

memmap_registry = {}


def del_temp():

    global TEMP_FOLDER

    try:
        close_all_memmap()
    except Exception as e:
        print("Impossible to close all memmaps")
        print(e)

    try:
        shutil.rmtree(TEMP_FOLDER)
        TEMP_FOLDER = tempfile.mkdtemp()
        print("All temporary files deleted.")
    except Exception as e:
        print("Impossible to clean temporary directory.")
        print(e)


def close_memmap(name):
    if memmap_registry.get(name) is not None:
        memmap_registry[name].close()


def close_all_memmap():
    for fp in memmap_registry.values():
        fp.close()


def create_empty_memmap(shape):
    filename = TEMP_FOLDER + f"/{uuid.uuid4()}"
    fp = open(filename, "wb+")
    memmap_registry[filename] = fp
    return np.memmap(fp, shape=shape, dtype=np.float64)


def create_memmap(data, name):

    if isinstance(data, dict):
        mmap_data = {}
        tmp = TEMP_FOLDER
        filename = tmp + f"/{name}.dat"
        file = open(filename, mode="wb+")

        item = np.asarray(data[list(data.keys())[0]])

        if item.ndim == 1:
            total_shape = (np.sum([np.asarray(d).shape[0] for d in data.values()]),)
        elif item.ndim == 2:
            total_shape = (
                np.sum([np.asarray(d).shape[0] for d in data.values()]),
                item.shape[1],
            )

        mmap = np.memmap(file, dtype=item.dtype, mode="w+", shape=total_shape)

        memmap_registry[name] = file

        start = 0
        for song, datum in data.items():
            datum = np.asarray(datum)
            end = datum.shape[0] + start
            mmap[start:end] = datum[:]
            mmap_data[song] = mmap[start:end]
            start = end

        return mmap_data

    elif isinstance(data, np.ndarray):
        tmp = tempfile.mkdtemp()
        filename = tmp + f"/{name}"
        dump(data, filename)
        return load(filename, mmap_mode="r+")

    else:
        warnings.warn("Impossible to create memmap.")
        return data


# found on https://www.scivision.dev/platform-independent-ulimit-number-of-open-files-increase/
def raise_nofile(nofile_atleast=None):
    """
    sets nofile soft limit to at least 4096, useful for running matlplotlib/seaborn on
    parallel executing plot generators vs. Ubuntu default ulimit -n 1024 or OS X El Captian 256
    temporary setting extinguishing with Python session.
    """
    try:
        import resource as res
    except ImportError:  # Windows
        return

    # %% (0) what is current ulimit -n setting?
    soft, ohard = res.getrlimit(res.RLIMIT_NOFILE)
    hard = ohard
    # %% (1) increase limit (soft and even hard) if needed
    if nofile_atleast is None:
        nofile_atleast = ohard

    if soft < nofile_atleast:
        soft = nofile_atleast

        if hard < soft:
            hard = soft

        print("Setting soft & hard ulimit -n {} {}".format(soft, hard))
        try:
            res.setrlimit(res.RLIMIT_NOFILE, (soft, hard))
        except (ValueError, res.error):
            try:
                hard = soft
                print(
                    "Trouble with max limit, retrying with soft,hard {},{}".format(
                        soft, hard
                    )
                )
                res.setrlimit(res.RLIMIT_NOFILE, (soft, hard))
            except Exception as e:
                print("Failed to set ulimit, giving up")
                soft, hard = res.getrlimit(res.RLIMIT_NOFILE)
                print(e)
