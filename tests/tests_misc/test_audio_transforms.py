# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
"""Tests for canapy/transforms/commons/audio.py — ls_audio_dir, ls_spec_dir, load_audio, get_filenames."""

import numpy as np
import pandas as pd
import pytest
import scipy.io.wavfile

from canapy.transforms.commons.audio import (
    ls_audio_dir,
    ls_spec_dir,
    load_audio,
    get_filenames,
)


class TestLsAudioDir:
    def test_finds_wav_files(self, tmp_path):
        (tmp_path / "a.wav").write_bytes(b"")
        (tmp_path / "b.wav").write_bytes(b"")
        result = ls_audio_dir(str(tmp_path), ".wav")
        assert len(result) == 2

    def test_ignores_other_extensions(self, tmp_path):
        (tmp_path / "audio.wav").write_bytes(b"")
        (tmp_path / "notes.txt").write_bytes(b"")
        result = ls_audio_dir(str(tmp_path), ".wav")
        assert len(result) == 1

    def test_empty_directory_returns_empty_list(self, tmp_path):
        result = ls_audio_dir(str(tmp_path), ".wav")
        assert result == []

    def test_returns_list(self, tmp_path):
        result = ls_audio_dir(str(tmp_path), ".wav")
        assert isinstance(result, list)

    def test_recursive_glob(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.wav").write_bytes(b"")
        result = ls_audio_dir(str(tmp_path), ".wav")
        assert len(result) == 1


class TestLsSpecDir:
    def test_nonexistent_directory_returns_empty_df(self, tmp_path):
        result = ls_spec_dir(str(tmp_path / "nope"), ".mfcc.npz")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_empty_directory_returns_empty_df(self, tmp_path):
        result = ls_spec_dir(str(tmp_path), ".mfcc.npz")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_npz_with_notated_path_field(self, tmp_path):
        path = tmp_path / "bird.mfcc.npz"
        np.savez(str(path), feature=np.zeros((13, 100)), notated_path="bird.wav")
        result = ls_spec_dir(str(tmp_path), ".mfcc.npz")
        assert len(result) == 1
        assert "notated_path" in result.columns
        assert "feature_path" in result.columns
        assert result.iloc[0]["notated_path"] == "bird.wav"

    def test_npz_without_notated_path_gives_nan(self, tmp_path):
        path = tmp_path / "bird.mfcc.npz"
        np.savez(str(path), feature=np.zeros((13, 50)))
        result = ls_spec_dir(str(tmp_path), ".mfcc.npz")
        assert len(result) == 1
        assert pd.isna(result.iloc[0]["notated_path"])

    def test_multiple_npz_files(self, tmp_path):
        for i in range(3):
            np.savez(
                str(tmp_path / f"s{i}.mfcc.npz"),
                feature=np.zeros((13, 50)),
                notated_path=f"s{i}.wav",
            )
        result = ls_spec_dir(str(tmp_path), ".mfcc.npz")
        assert len(result) == 3


class TestLoadAudio:
    def test_loads_wav_returns_1d(self, tmp_path):
        rate = 22050
        wav = (np.random.default_rng(0).random(rate) * 2 - 1)
        wav_i16 = (wav * np.iinfo(np.int16).max).astype(np.int16)
        path = tmp_path / "audio.wav"
        scipy.io.wavfile.write(str(path), rate, wav_i16)
        audio = load_audio(str(path), sampling_rate=rate)
        assert audio.ndim == 1
        assert len(audio) > 0

    def test_loads_npy_1d(self, tmp_path):
        arr = np.random.default_rng(0).random(1000).astype(np.float32)
        path = tmp_path / "audio.npy"
        np.save(str(path), arr)
        audio = load_audio(str(path), sampling_rate=22050)
        assert audio.ndim == 1
        assert len(audio) == 1000

    def test_loads_npy_2d_channels_first_converts_to_mono(self, tmp_path):
        # shape (2, 1000) — two channels in first axis
        arr = np.random.default_rng(0).random((2, 1000)).astype(np.float32)
        path = tmp_path / "stereo.npy"
        np.save(str(path), arr)
        audio = load_audio(str(path), sampling_rate=22050)
        assert audio.ndim == 1

    def test_loads_npy_2d_time_first_converts_to_mono(self, tmp_path):
        # shape (1000, 2) — time first, two channels
        arr = np.random.default_rng(0).random((1000, 2)).astype(np.float32)
        path = tmp_path / "stereo_tf.npy"
        np.save(str(path), arr)
        audio = load_audio(str(path), sampling_rate=22050)
        assert audio.ndim == 1

    def test_npy_3d_raises_value_error(self, tmp_path):
        arr = np.zeros((4, 4, 4), dtype=np.float32)
        path = tmp_path / "bad.npy"
        np.save(str(path), arr)
        with pytest.raises(ValueError):
            load_audio(str(path), sampling_rate=22050)


class TestGetFilenames:
    def test_extracts_stem_without_extension(self):
        result = get_filenames(["/path/to/bird_01.wav", "other/bird_02.wav"])
        assert result == ["bird_01", "bird_02"]

    def test_strips_first_extension_of_double_ext(self):
        # "bird_01.mfcc.npz" → stem is "bird_01.mfcc", split(".")[0] → "bird_01"
        result = get_filenames(["/path/to/bird_01.mfcc.npz"])
        assert result == ["bird_01"]

    def test_empty_list(self):
        assert get_filenames([]) == []

    def test_single_file(self):
        result = get_filenames(["song.wav"])
        assert result == ["song"]

    def test_preserves_order(self):
        paths = ["c.wav", "a.wav", "b.wav"]
        result = get_filenames(paths)
        assert result == ["c", "a", "b"]
