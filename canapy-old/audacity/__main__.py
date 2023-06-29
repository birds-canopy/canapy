"""
A simple tool to convert Audacity projects with audio annotations to .wav files and .csv comma-separated dataset.
The resulting dataset contains five features per annotations:

  - a "wave" column storing the name of the wave file containing the annotation,

  - a "syll" column storing the annotation label,

  - a "start" colum storing the start time of the annotation on the track in seconds,

  - a "end" column storing its end time, also in seconds.

Also creates a repertory with mel-spectrogam representations of the annotations, and individual .wav files for
each annotation.
"""
import os
import json
import math
import xml.etree.ElementTree as ET
import argparse

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import librosa as lbr
import soundfile as sf
import seaborn as sns
from tqdm import tqdm

from . import aup as audacity
from ..dataset import Config
from ..config import default_config

sns.set(context="paper", style="dark")

AUP_HEAD = (
    '<?xml version="1.0" standalone="no" ?> '
    '<!DOCTYPE project PUBLIC "-//audacityproject-1.3.0//DTD//EN" '
    '"http://audacity.sourceforge.net/xml/audacityproject-1.3.0.dtd" >'
)

_epilog = """
Note: if no configuration file is given, the script will use a default configuration file.

\nInformation needed in this configuration file is:

\n\t-sampling_rate: sampling rate to apply to wave files,

\n\t-n_fft, hop_length: FFT window length and strides (in seconds).
"""

parser = argparse.ArgumentParser(description=__doc__, epilog=_epilog)

parser.add_argument(
    "datasource",
    type=str,
    action="extend",
    nargs="+",
    help="Directories containing .aup file and Audacity data directories.",
)
parser.add_argument(
    "--repertoire",
    "-r",
    action="store_true",
    default=False,
    help="Build a repertoire storing copies of all annotations in the form "
    "of isolated .wav files and .png spectrograms.",
)
parser.add_argument(
    "--output",
    "-o",
    type=str,
    nargs=1,
    required=True,
    help="Destination directory for the extracted data.",
)
parser.add_argument(
    "--config",
    "-c",
    type=str,
    nargs=1,
    help="JSON configuration file containing informations about data preprocessing.",
)


def fetch_audacity_projects(datasource):
    """Find all Audacity projects directories and files.

    Parameters
    ----------
    datasource : str or Path-like object
        Source directory containing the projects.

    Returns
    -------
    list of str or Path-like objects
        All found files
    """
    all_files = []
    for directory in datasource:
        directory = Path(directory)
        try:
            if directory.exists():
                files = [f for f in directory.iterdir() if f.suffix == ".aup"]

                if len(files) != 0:
                    print(
                        f"Found {len(files)} Audacity .aup projects files in {directory}."
                    )
                    all_files.extend(files)
                else:
                    raise FileNotFoundError(
                        f"No Audacity .aup project files found in {directory}."
                    )
            else:
                raise NotADirectoryError(f"Directory {directory} not found.")
        except Exception as e:
            print(f"WARNING caught exception: {e}")

    if len(all_files) < 1:
        raise FileNotFoundError(
            f"No Audacity .aup project files found in {datasource}."
        )

    return all_files


def get_config(config_path):
    """Get user defined JSON configuration file,
    or return the default config.

    Parameters
    ----------
    config_path : str or Path-like object, optional
        Configuration file to open. Defaults to None.

    Returns
    -------
    Config
        A Config object storing the configuration.
    """
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.is_file():
            with config_path.open("r") as f:
                config = Config(json.load(f))
                check_config(config)
            return config
        else:
            raise FileNotFoundError(f"No JSON configuration file {config_path} found.")
    else:
        return Config(default_config())


def check_config(config):
    """Check the presence of required args in
    configuration.

    Required arguments for Audacity projects exportation
    are 'n_fft', 'hop_length' and 'sampling_rate'.

    Parameters
    ----------
    config : Config
    """
    for arg in ["n_fft", "hop_length", "sampling_rate"]:
        if config.get(arg) is None:
            raise ValueError(
                f"Argument '{arg}' is required but not found in config file."
            )


def build_output_directories(out, repertoire=True):
    """Build directory structure for dataset.

    Parameters
    ----------
    out : str or Path like object
        Root for the dataset directories
    repertoire : bool, optional
        If True, will also build directories to
        store separate copies of each repertoire
        sample in the dataset.
    Returns
    -------
    Path like objects
        Root directory, audio directory, annotations directory
        and optionally repertoire directory.
    """
    out = Path(out)
    if not out.exists():
        out.mkdir(parents=True)

    audios_dir = out / "data"
    annots_dir = out / "data"
    rep_dir = None

    if not audios_dir.exists():
        audios_dir.mkdir(parents=True)

    if repertoire:
        rep_dir = out / "repertoire"
        if not rep_dir.exists():
            rep_dir.mkdir(parents=True)

    return out, audios_dir, annots_dir, rep_dir


def are_unique(objs):
    """Asserts if duplicates exist in
    a sequence of objects."""
    seen = set()
    for x in objs:
        if x not in seen:
            seen.add(x)
        else:
            return False
    return True


def list_waves(files):
    """List all .wav files in Audacity project .aup files."""
    waves = []
    for file in files:
        ET.register_namespace("", "http://audacity.sourceforge.net/xml/")
        tree = ET.parse(file)
        root = tree.getroot()
        for w in root.findall("{http://audacity.sourceforge.net/xml/}wavetrack"):
            waves.append(w.get("name"))
    return waves


def prepend_line(file_name, line):
    """Insert given string as a new line at the beginning of a file"""
    dummy_file = file_name + ".tmp"
    with open(file_name, "r") as read_obj, open(dummy_file, "w") as write_obj:
        write_obj.write(line + "\n")
        for line in read_obj:
            write_obj.write(line)
    os.remove(file_name)
    os.rename(dummy_file, file_name)


def extract_data(files):
    """Convert Audacity .aup files with annotations
    to pandas.DataFrame storing all annotations.

    This conversion may change some settings in the
    Audacity projects, like resetting audio gain to
    1.0 or attaching unique ID to each track. A new version
    of the projects will therefore be saved. Original projects
    will be moved to an 'audacity_old' directory.

    Parameters
    ----------
    files : list of str or Path like objects
        Audacity .aup projects.

    Returns
    -------
    pandas.DataFrame
        All annotations as a dataframe.
    """
    all_waves = list_waves(files)

    # check if all tracks in the audacity project
    # have unique IDs. If not, automatically prepend
    # an unique ID to each track name.
    update_trackname = False
    if not (are_unique(all_waves)):
        track_id = 0
        update_trackname = True

    rows = []
    for file in tqdm(files, "Extracting labels from project"):
        ET.register_namespace("", "http://audacity.sourceforge.net/xml/")
        tree = ET.parse(file)
        root = tree.getroot()

        # all nodes representing .wav files
        waves = root.findall("{http://audacity.sourceforge.net/xml/}wavetrack")
        # all nodes representing annotation track
        labeltrack = root.findall("{http://audacity.sourceforge.net/xml/}labeltrack")

        for wave, labels in zip(waves, labeltrack):

            wave_name = wave.get("name")
            if update_trackname:
                # prepend an unique ID to track name
                wave_name = f"{str(track_id).zfill(3)}-{wave_name}"
                wave.set("name", wave_name)
                labels.set("name", f"lbl-{wave_name}")
                track_id += 1

            # reset gain
            wave.set("gain", str(1.0))

            # build the dataset
            for lbl in labels.findall("{http://audacity.sourceforge.net/xml/}label"):
                rows.append(
                    pd.DataFrame(
                        [
                            {
                                "wave": wave_name + ".wav",
                                "start": lbl.get("t"),
                                "end": lbl.get("t1"),
                                "syll": lbl.get("title"),
                            }
                        ]
                    )
                )

        # save new Audacity projects, with unique IDs for each track
        # and without any gain.
        # In order to save the files as Audacity projects, must prepend
        # Audacity XML header to each file manually.
        filename = os.path.splitext(file)[0]
        tree.write(filename + ".tmp")
        prepend_line(filename + ".tmp", AUP_HEAD)

        # Original projects are moved to an 'audacity_old' directory.
        olddir = os.path.join(os.path.dirname(file), "audacity_old")
        if not (os.path.isdir(olddir)):
            os.mkdir(olddir)
            os.rename(file, os.path.join(olddir, os.path.split(file)[1]))
            os.rename(filename + ".tmp", filename + ".aup")

    df = pd.concat(rows, ignore_index=True)

    df["start"] = pd.to_numeric(df["start"])
    df["end"] = pd.to_numeric(df["end"])
    df = df.sort_values(by=["wave", "start"], axis=0, ascending=True)
    df.reset_index(drop=True, inplace=True)

    print(f"{len(df.groupby('wave').groups.keys())} annotations tracks extracted.")

    return df


def extract_audio(files, audio_dir):
    """Save audio tracks from Audacity .aup project files
    to a specific directory.

    All audio tracks are saved as separated .wav files.

    Parameters
    ----------
    files : list of str or Path like objects
        All Audacity projects to extract.
    audio_dir : str or Path like object
        Destination directory for audio files.
    """
    success = 0
    failure = 0
    file_failure = 0
    for file in files:
        try:
            aup = audacity.Aup(file)
            all_waves = list_waves([file])

            for c, w in tqdm(
                enumerate(all_waves),
                f"Exporting audio from project {file}",
                total=len(all_waves),
            ):
                try:
                    aup.towav(str(audio_dir / f"{w}.wav"), c)
                except Exception as e:
                    print(e)
                    failure += 1

                success += 1

        except Exception as e:
            print(e)
            file_failure += 1

    print(
        f"{success} audio tracks sucessfuly extracted. "
        f"{failure} audio extraction failures. "
        f"{file_failure} Audacity file failure."
    )


def get_save_path(syll, start, end, sr, audio_file):
    file_name = f"{audio_file.split('.')[0]}"
    start, end = math.ceil(start * sr), math.ceil(end * sr)
    file_name = f"{syll}-{start}-{end}-{file_name}"
    return file_name


def cut_sample(audio, mel, start, end, config):
    """Extract a phrase sample from a song.

    Parameters
    ----------
    audio : np.ndarray
        Song audio track
    mel : np.ndarray
        Mel spectrogram of the track
    start : float
        Start of phrase
    end : float
        End of phrase
    config : Config
        Configuration object

    Returns
    -------
    np.ndarray, np.ndarray, float
        [description]
    """
    sr = config.sampling_rate
    mr = sr / config.as_fftwindow("hop_length")
    dilatation = 1 / 25
    delta = dilatation * 1 / (end - start)
    wstart = math.ceil(start * sr)
    wend = math.ceil(end * sr)
    mstart = math.ceil(start * mr)
    mend = math.ceil(end * mr)

    # if (start-delta >= 0) and (end+delta) < (len(audio)/sr):
    #     wstart = math.ceil((start - delta) * sr)
    #     wend = math.ceil((end + delta) * sr)

    sample = audio[wstart:wend]

    if (start - delta) * mr >= 0 and (end + delta) * mr < mel.shape[1]:
        mstart = math.ceil((start - delta) * mr)
        mend = math.ceil((end + delta) * mr)

    mel_sample = mel[:, mstart:mend]

    return sample, mel_sample, delta


def export_repertoire(df, audio_dir, rep_dir, config):

    waves = df.groupby("wave").groups.keys()

    df["repertoire_file"] = [None for i in range(len(df))]

    success = 0
    failure = 0
    for file in tqdm(waves, "Exporting repertoire"):

        wave, sr = lbr.load(audio_dir / file, sr=config.sampling_rate)
        mel = lbr.power_to_db(
            lbr.feature.melspectrogram(wave, sr, n_fft=1024, hop_length=512)
        )
        lbl = df.loc[df["wave"] == file]

        mr = config.sampling_rate / config.as_fftwindow("hop_length")

        for lbl_entry in lbl.itertuples():
            index = lbl_entry.Index
            syll = lbl_entry.syll
            start = lbl_entry.start
            end = lbl_entry.end

            if syll != "SIL":
                try:
                    sample, mel_sample, delta = cut_sample(
                        wave, mel, start, end, config
                    )
                    file_name = get_save_path(syll, start, end, sr, file)

                    fig = plt.figure()
                    ax = plt.gca()
                    plt.axis("off")
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

                    ax.imshow(mel_sample, origin="lower", cmap="magma")
                    ax.axvline(math.floor(delta * mr), color="white", linestyle="--")
                    ax.axvline(
                        math.ceil(mel_sample.shape[1] - delta * mr),
                        color="white",
                        linestyle="--",
                    )

                    fig.savefig(f"{Path(rep_dir) / file_name}.png")
                    plt.clf()
                    plt.close(fig)

                    sf.write(f"{Path(rep_dir) / file_name}.wav", sample, sr)

                    df.at[index, "repertoire_file"] = file_name

                    success += 1

                except Exception as e:
                    failure += 1
                    print(f"Repertoire failure : syllable {syll}: {e}")

    print(f"{success} syllables sucessfuly exported in repertoire. {failure} failures.")


def export_annotations(df, annots_dir):
    """Save all annotations a CSV files.

    Parameters
    ----------
    df : pandas.Dataframe
        Dataframe to save.
    annots_dir : [type]
        Output directory for the
    """
    songs = df.groupby("wave").groups.keys()
    for song in tqdm(songs, f"Exporting labels from project"):
        labels = df[df["wave"] == song]
        out_file = annots_dir / f"{song.split('.')[0]}.csv"
        labels.to_csv(out_file, index=False)


def print_repertoire(df):
    """Print a summary of repertoire syllables and durations

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset.
    """
    df["duration"] = df["end"] - df["start"]

    # all counts
    syllables_counts = pd.DataFrame(df.groupby("syll")["end"].count())
    syllables_counts.columns = ["nb samples"]

    # all durations
    syllables_dur = pd.DataFrame(df.groupby("syll")["duration"].sum())
    syllables_dur.columns = ["duration"]

    # total
    syllables_counts["total duration (s)"] = syllables_dur["duration"]
    syllables_counts.sort_values(
        by=["total duration (s)"], ascending=False, inplace=True
    )

    print("Current repertoire :")
    print(syllables_counts.to_string())

    # clean up
    df.drop(["duration"], axis=1, inplace=True)


def convert(datasource=None, repertoire=None, config=None, output=None):

    config = get_config(config)

    files = fetch_audacity_projects(datasource)

    df = extract_data(files)

    print_repertoire(df)

    if input("Is it ok ? (y/n)\n") != "n":
        output, audio_dir, annots_dir, rep_dir = build_output_directories(
            output[0], repertoire
        )

        extract_audio(files, audio_dir)

        if repertoire:
            export_repertoire(df, audio_dir, rep_dir, config)

        export_annotations(df, annots_dir)

    else:
        print("Aborting.")
        return 0

    return 0


if __name__ == "__main__":

    args = parser.parse_args()

    convert(**vars(args))
