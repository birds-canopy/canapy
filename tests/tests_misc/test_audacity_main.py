# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
"""Tests for audacity/main.py."""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_NS = "http://audacity.sourceforge.net/xml/"


def _make_aup(tmp_path, track_names=None, filename="project.aup"):
    root = ET.Element(f"{{{_NS}}}project")
    root.set("rate", "44100")
    root.set("projname", "test")
    for name in (track_names or []):
        wt = ET.SubElement(root, f"{{{_NS}}}wavetrack")
        wt.set("name", name)
    path = tmp_path / filename
    ET.ElementTree(root).write(str(path), xml_declaration=True, encoding="unicode")
    return path


# ---------------------------------------------------------------------------
# are_unique
# ---------------------------------------------------------------------------

class TestAreUnique:
    def test_all_unique(self):
        from audacity.main import are_unique
        assert are_unique(["a", "b", "c"]) is True

    def test_with_duplicate(self):
        from audacity.main import are_unique
        assert are_unique(["a", "b", "a"]) is False

    def test_empty_sequence(self):
        from audacity.main import are_unique
        assert are_unique([]) is True

    def test_single_element(self):
        from audacity.main import are_unique
        assert are_unique(["x"]) is True


# ---------------------------------------------------------------------------
# build_output_directories
# ---------------------------------------------------------------------------

class TestBuildOutputDirectories:
    def test_creates_root_and_data_dirs(self, tmp_path):
        from audacity.main import build_output_directories
        root, audio_dir, annots_dir, rep_dir = build_output_directories(tmp_path / "output")
        assert root.exists()
        assert audio_dir.exists()
        assert annots_dir.exists()
        assert rep_dir is None

    def test_idempotent_when_dirs_exist(self, tmp_path):
        from audacity.main import build_output_directories
        out = tmp_path / "output"
        build_output_directories(out)
        build_output_directories(out)

    def test_data_dir_is_subdir_of_root(self, tmp_path):
        from audacity.main import build_output_directories
        root, audio_dir, _, _ = build_output_directories(tmp_path / "output")
        assert str(audio_dir).startswith(str(root))


# ---------------------------------------------------------------------------
# get_save_path
# ---------------------------------------------------------------------------

class TestGetSavePath:
    def test_contains_syll_and_audio_stem(self):
        from audacity.main import get_save_path
        result = get_save_path("A", 0.1, 0.2, 22050, "song.wav")
        assert "A" in result
        assert "song" in result

    def test_different_syllables_give_different_names(self):
        from audacity.main import get_save_path
        r1 = get_save_path("A", 0.1, 0.2, 22050, "song.wav")
        r2 = get_save_path("B", 0.1, 0.2, 22050, "song.wav")
        assert r1 != r2

    def test_different_timings_give_different_names(self):
        from audacity.main import get_save_path
        r1 = get_save_path("A", 0.1, 0.2, 22050, "song.wav")
        r2 = get_save_path("A", 0.5, 0.9, 22050, "song.wav")
        assert r1 != r2


# ---------------------------------------------------------------------------
# fetch_audacity_projects
# ---------------------------------------------------------------------------

class TestFetchAudacityProjects:
    def test_empty_directory_raises(self, tmp_path):
        from audacity.main import fetch_audacity_projects
        with pytest.raises(FileNotFoundError):
            fetch_audacity_projects([str(tmp_path)])

    def test_nonexistent_directory_raises(self):
        from audacity.main import fetch_audacity_projects
        with pytest.raises(FileNotFoundError):
            fetch_audacity_projects(["/this/path/does/not/exist/xyz"])

    def test_finds_aup_files(self, tmp_path):
        from audacity.main import fetch_audacity_projects
        (tmp_path / "project.aup").write_text("<project/>")
        files = fetch_audacity_projects([str(tmp_path)])
        assert len(files) == 1
        assert files[0].suffix == ".aup"

    def test_ignores_non_aup_files(self, tmp_path):
        from audacity.main import fetch_audacity_projects
        (tmp_path / "project.aup").write_text("<project/>")
        (tmp_path / "other.txt").write_text("ignored")
        files = fetch_audacity_projects([str(tmp_path)])
        assert all(f.suffix == ".aup" for f in files)


# ---------------------------------------------------------------------------
# export_annotations
# ---------------------------------------------------------------------------

class TestExportAnnotations:
    def test_creates_one_csv_per_wave(self, tmp_path):
        from audacity.main import export_annotations
        df = pd.DataFrame({
            "wave": ["bird_a.wav", "bird_a.wav", "bird_b.wav"],
            "start": [0.0, 0.1, 0.0],
            "end":   [0.1, 0.2, 0.1],
            "syll":  ["A", "B", "A"],
        })
        export_annotations(df, tmp_path)
        assert len(list(tmp_path.glob("*.csv"))) == 2

    def test_csv_content_matches_input(self, tmp_path):
        from audacity.main import export_annotations
        df = pd.DataFrame({
            "wave":  ["x.wav", "x.wav"],
            "start": [0.0, 0.5],
            "end":   [0.5, 1.0],
            "syll":  ["A", "B"],
        })
        export_annotations(df, tmp_path)
        loaded = pd.read_csv(tmp_path / "x.csv")
        assert list(loaded["syll"]) == ["A", "B"]


# ---------------------------------------------------------------------------
# print_repertoire
# ---------------------------------------------------------------------------

class TestPrintRepertoire:
    def test_does_not_crash(self, capsys):
        from audacity.main import print_repertoire
        df = pd.DataFrame({
            "wave":  ["x.wav"] * 4,
            "start": [0.0, 0.1, 0.2, 0.3],
            "end":   [0.1, 0.2, 0.3, 0.4],
            "syll":  ["A", "B", "A", "C"],
        })
        print_repertoire(df)
        out = capsys.readouterr().out
        assert "A" in out

    def test_drops_duration_column_after_call(self):
        from audacity.main import print_repertoire
        df = pd.DataFrame({
            "wave":  ["x.wav"] * 2,
            "start": [0.0, 0.1],
            "end":   [0.1, 0.2],
            "syll":  ["A", "B"],
        })
        print_repertoire(df)
        assert "duration" not in df.columns


# ---------------------------------------------------------------------------
# list_waves
# ---------------------------------------------------------------------------

class TestListWaves:
    def test_no_wavetracks_returns_empty_list(self, tmp_path):
        from audacity.main import list_waves
        aup = _make_aup(tmp_path, track_names=[])
        result = list_waves([aup])
        assert result == []

    def test_single_wavetrack_name(self, tmp_path):
        from audacity.main import list_waves
        aup = _make_aup(tmp_path, track_names=["Song A"])
        result = list_waves([aup])
        assert result == ["Song A"]

    def test_multiple_wavetrack_names(self, tmp_path):
        from audacity.main import list_waves
        aup = _make_aup(tmp_path, track_names=["Track 1", "Track 2", "Track 3"])
        result = list_waves([aup])
        assert result == ["Track 1", "Track 2", "Track 3"]

    def test_multiple_aup_files(self, tmp_path):
        from audacity.main import list_waves
        aup1 = _make_aup(tmp_path, track_names=["A"], filename="p1.aup")
        aup2 = _make_aup(tmp_path, track_names=["B", "C"], filename="p2.aup")
        result = list_waves([aup1, aup2])
        assert result == ["A", "B", "C"]

    def test_empty_file_list(self):
        from audacity.main import list_waves
        assert list_waves([]) == []


# ---------------------------------------------------------------------------
# prepend_line
# ---------------------------------------------------------------------------

class TestPrependLine:
    def test_prepended_line_is_first(self, tmp_path):
        from audacity.main import prepend_line
        path = str(tmp_path / "file.txt")
        Path(path).write_text("line2\nline3\n")
        prepend_line(path, "line1")
        lines = Path(path).read_text().splitlines()
        assert lines[0] == "line1"

    def test_original_content_preserved(self, tmp_path):
        from audacity.main import prepend_line
        path = str(tmp_path / "file.txt")
        Path(path).write_text("original\n")
        prepend_line(path, "new")
        assert "original" in Path(path).read_text()

    def test_file_still_exists_after_prepend(self, tmp_path):
        from audacity.main import prepend_line
        path = str(tmp_path / "file.txt")
        Path(path).write_text("data\n")
        prepend_line(path, "header")
        assert os.path.exists(path)

    def test_no_tmp_file_left_behind(self, tmp_path):
        from audacity.main import prepend_line
        path = str(tmp_path / "file.txt")
        Path(path).write_text("content\n")
        prepend_line(path, "start")
        assert not os.path.exists(path + ".tmp")

    def test_prepend_to_empty_file(self, tmp_path):
        from audacity.main import prepend_line
        path = str(tmp_path / "empty.txt")
        Path(path).write_text("")
        prepend_line(path, "first")
        assert Path(path).read_text().startswith("first")
