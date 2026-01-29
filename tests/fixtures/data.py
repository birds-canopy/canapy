import shutil
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
import scipy.io


@pytest.fixture(scope='session')
def audio_directory(tmpdir_factory):
    audio_dir = Path(str(tmpdir_factory.mktemp("audio")))
    rate = 44100

    for i in range(50):
        length = np.random.randint(99999, 999999)
        signal = np.random.uniform(-1, 1, length)
        
        wave = np.int16(signal / np.max(np.abs(signal)) * np.iinfo(np.int16).max)
        scipy.io.wavfile.write(str(audio_dir / f"{str(i).zfill(2)}-fake.wav"), rate, wave)

    yield audio_dir

    shutil.rmtree(str(audio_dir))


@pytest.fixture(scope='session')
def annot_directory(tmpdir_factory, audio_directory):
    annot_dir = Path(str(tmpdir_factory.mktemp("annots")))
    labels = list("abcdefgh")
    
    for file in Path(audio_directory).glob("*.wav"):
        
        rate, audio = scipy.io.wavfile.read(str(file))
        
        length = len(audio) / rate
        partitions = 2*np.random.randint(1, length)
    
        bounds = np.sort(np.random.uniform(0.0, length, size=partitions))
        onsets = np.concatenate([np.array([0.0]), bounds])
        offsets = np.concatenate([bounds, np.array([length])])
        
        annotations = []
        for onset, offset in zip(onsets, offsets):
            if np.random.rand() > 0.5:
                annotations.append(
                    {
                        "wave": str(file),
                        "start": onset,
                        "end": offset,
                        "syll": np.random.choice(labels)
                    }
                )
        
        if len(annotations) == 0:
            annotations = [
                {
                    "wave": str(file),
                    "start": 0.0,
                    "end": length,
                    "syll": "SIL",
                }
            ]

        annotations = pd.DataFrame(annotations)
        annotations.to_csv(str(annot_dir / f"{file.stem}.annot.csv"), index=False)

    yield annot_dir
    
    shutil.rmtree(str(annot_dir))


@pytest.fixture()
def output_directory(tmpdir_factory):
    output_dir = Path(str(tmpdir_factory.mktemp("output")))
    yield output_dir
    shutil.rmtree(str(output_dir))


@pytest.fixture()
def spec_directory(tmpdir_factory):
    spec_dir = Path(str(tmpdir_factory.mktemp("spec")))
    yield spec_dir
    shutil.rmtree(str(spec_dir))
