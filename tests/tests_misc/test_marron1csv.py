# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
"""Tests for canapy/formats/marron1csv.py — Marron1CSV format."""

import numpy as np
import pandas as pd
import pytest


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


class TestMarron1CSVFromFile:
    def test_basic_read(self, tmp_path):
        from canapy.formats.marron1csv import Marron1CSV
        p = tmp_path / "a.csv"
        _write_csv(p, [
            {"wave": "bird.wav", "start": 0.0, "end": 0.1, "syll": "A"},
            {"wave": "bird.wav", "start": 0.2, "end": 0.3, "syll": "B"},
        ])
        m = Marron1CSV.from_file(p)
        assert len(m.starts) == 2
        assert len(m.ends) == 2
        assert len(m.sylls) == 2

    def test_notated_path_contains_wave_name(self, tmp_path):
        from canapy.formats.marron1csv import Marron1CSV
        p = tmp_path / "a.csv"
        _write_csv(p, [{"wave": "bird.wav", "start": 0.0, "end": 0.1, "syll": "A"}])
        m = Marron1CSV.from_file(p)
        assert "bird.wav" in str(m.notated_path)

    def test_multiple_audio_files_raises(self, tmp_path):
        from canapy.formats.marron1csv import Marron1CSV, DataFormatError
        p = tmp_path / "a.csv"
        _write_csv(p, [
            {"wave": "bird_a.wav", "start": 0.0, "end": 0.1, "syll": "A"},
            {"wave": "bird_b.wav", "start": 0.1, "end": 0.2, "syll": "B"},
        ])
        with pytest.raises(DataFormatError):
            Marron1CSV.from_file(p)

    def test_wrong_extension_raises(self, tmp_path):
        from canapy.formats.marron1csv import Marron1CSV
        p = tmp_path / "a.txt"
        p.write_text("wave,start,end,syll\nbird.wav,0,1,A\n")
        with pytest.raises(Exception):
            Marron1CSV.from_file(p)

    def test_nullable_syll_accepted(self, tmp_path):
        from canapy.formats.marron1csv import Marron1CSV
        p = tmp_path / "a.csv"
        _write_csv(p, [{"wave": "bird.wav", "start": 0.0, "end": 0.1, "syll": None}])
        m = Marron1CSV.from_file(p)
        assert len(m.starts) == 1

    def test_notated_path_override(self, tmp_path):
        from canapy.formats.marron1csv import Marron1CSV
        p = tmp_path / "a.csv"
        _write_csv(p, [{"wave": "bird.wav", "start": 0.0, "end": 0.1, "syll": "A"}])
        m = Marron1CSV.from_file(p, notated_path=tmp_path)
        assert str(tmp_path) in str(m.notated_path)


class TestMarron1CSVToSeq:
    def _make(self, tmp_path):
        from canapy.formats.marron1csv import Marron1CSV
        p = tmp_path / "a.csv"
        _write_csv(p, [
            {"wave": "bird.wav", "start": 0.0,      "end": 0.1,      "syll": "A"},
            {"wave": "bird.wav", "start": 0.123456, "end": 0.234567, "syll": "B"},
        ])
        return Marron1CSV.from_file(p)

    def test_returns_crowsetta_sequence(self, tmp_path):
        import crowsetta
        m = self._make(tmp_path)
        assert isinstance(m.to_seq(), crowsetta.Sequence)

    def test_segment_count(self, tmp_path):
        m = self._make(tmp_path)
        assert len(m.to_seq().segments) == 2

    def test_rounding(self, tmp_path):
        m = self._make(tmp_path)
        seq = m.to_seq(round_times=True, decimals=3)
        assert seq.segments[1].onset_s == pytest.approx(0.123, abs=1e-3)

    def test_no_rounding(self, tmp_path):
        m = self._make(tmp_path)
        seq = m.to_seq(round_times=False)
        assert seq.segments[1].onset_s == pytest.approx(0.123456, abs=1e-6)


class TestMarron1CSVToAnnot:
    def test_returns_crowsetta_annotation(self, tmp_path):
        import crowsetta
        from canapy.formats.marron1csv import Marron1CSV
        p = tmp_path / "a.csv"
        _write_csv(p, [{"wave": "bird.wav", "start": 0.0, "end": 0.1, "syll": "A"}])
        m = Marron1CSV.from_file(p)
        assert isinstance(m.to_annot(), crowsetta.Annotation)

    def test_annot_path_preserved(self, tmp_path):
        from canapy.formats.marron1csv import Marron1CSV
        p = tmp_path / "a.csv"
        _write_csv(p, [{"wave": "bird.wav", "start": 0.0, "end": 0.1, "syll": "A"}])
        m = Marron1CSV.from_file(p)
        annot = m.to_annot()
        assert annot.annot_path == p


class TestMarron1CSVToFile:
    def test_roundtrip(self, tmp_path):
        from canapy.formats.marron1csv import Marron1CSV
        src = tmp_path / "src.csv"
        _write_csv(src, [
            {"wave": "bird.wav", "start": 0.0, "end": 0.1, "syll": "A"},
            {"wave": "bird.wav", "start": 0.2, "end": 0.3, "syll": "B"},
        ])
        m = Marron1CSV.from_file(src)
        dst = tmp_path / "dst.csv"
        m.to_file(dst)
        m2 = Marron1CSV.from_file(dst)
        np.testing.assert_array_equal(m.starts, m2.starts)
        np.testing.assert_array_equal(m.ends, m2.ends)
        np.testing.assert_array_equal(m.sylls, m2.sylls)

    def test_output_file_has_required_columns(self, tmp_path):
        from canapy.formats.marron1csv import Marron1CSV
        src = tmp_path / "src.csv"
        _write_csv(src, [{"wave": "bird.wav", "start": 0.0, "end": 0.1, "syll": "A"}])
        m = Marron1CSV.from_file(src)
        dst = tmp_path / "dst.csv"
        m.to_file(dst)
        df = pd.read_csv(dst)
        for col in ("wave", "start", "end", "syll"):
            assert col in df.columns
