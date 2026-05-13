# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
"""Tests for audacity/aup.py — classes Aup and Channel."""

import xml.etree.ElementTree as ET

import pytest


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_aup_xml(tmp_path, rate=22050, projname="testproj", n_wavetracks=0):
    """Crée un fichier .aup XML minimal avec N wavetracks vides (sans blocs audio)."""
    ns = "http://audacity.sourceforge.net/xml/"
    root = ET.Element(f"{{{ns}}}project")
    root.set("rate", str(rate))
    root.set("projname", projname)
    for _ in range(n_wavetracks):
        ET.SubElement(root, f"{{{ns}}}wavetrack")
    aup_path = tmp_path / "test.aup"
    ET.ElementTree(root).write(
        str(aup_path), xml_declaration=True, encoding="unicode"
    )
    return aup_path


# ---------------------------------------------------------------------------
# Aup
# ---------------------------------------------------------------------------

class TestAup:
    def test_init_no_wavetracks(self, tmp_path):
        from audacity.aup import Aup
        aup = Aup(str(_make_aup_xml(tmp_path, n_wavetracks=0)))
        assert aup.nchannels == 0
        assert aup.rate == 22050

    def test_init_one_empty_wavetrack(self, tmp_path):
        from audacity.aup import Aup
        aup = Aup(str(_make_aup_xml(tmp_path, n_wavetracks=1)))
        assert aup.nchannels == 1
        assert aup.files[0] == []

    def test_init_two_empty_wavetracks(self, tmp_path):
        from audacity.aup import Aup
        aup = Aup(str(_make_aup_xml(tmp_path, n_wavetracks=2)))
        assert aup.nchannels == 2

    def test_custom_rate(self, tmp_path):
        from audacity.aup import Aup
        aup = Aup(str(_make_aup_xml(tmp_path, rate=44100, n_wavetracks=0)))
        assert aup.rate == 44100

    def test_open_returns_channel(self, tmp_path):
        from audacity.aup import Aup, Channel
        aup = Aup(str(_make_aup_xml(tmp_path, n_wavetracks=1)))
        ch = aup.open(0)
        assert isinstance(ch, Channel)

    def test_open_invalid_channel_raises_value_error(self, tmp_path):
        from audacity.aup import Aup
        aup = Aup(str(_make_aup_xml(tmp_path, n_wavetracks=1)))
        with pytest.raises(ValueError):
            aup.open(5)

    def test_open_negative_channel_raises_value_error(self, tmp_path):
        from audacity.aup import Aup
        aup = Aup(str(_make_aup_xml(tmp_path, n_wavetracks=1)))
        with pytest.raises(ValueError):
            aup.open(-1)

    def test_context_manager_does_not_raise(self, tmp_path):
        from audacity.aup import Aup
        with Aup(str(_make_aup_xml(tmp_path, n_wavetracks=0))) as aup:
            assert aup.nchannels == 0


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------

class TestChannel:
    def _open_channel(self, tmp_path):
        from audacity.aup import Aup
        aup = Aup(str(_make_aup_xml(tmp_path, n_wavetracks=1)))
        return aup.open(0)

    def test_initial_aunr_is_zero(self, tmp_path):
        ch = self._open_channel(tmp_path)
        assert ch.aunr == 0

    def test_close_sets_aunr_negative(self, tmp_path):
        ch = self._open_channel(tmp_path)
        ch.close()
        assert ch.aunr == -1

    def test_seek_on_closed_channel_raises_io_error(self, tmp_path):
        ch = self._open_channel(tmp_path)
        ch.close()
        with pytest.raises(IOError):
            ch.seek(0)

    def test_read_on_closed_channel_raises_io_error(self, tmp_path):
        ch = self._open_channel(tmp_path)
        ch.close()
        with pytest.raises(IOError):
            list(ch.read())

    def test_read_empty_wavetrack_yields_nothing(self, tmp_path):
        ch = self._open_channel(tmp_path)
        assert list(ch.read()) == []

    def test_seek_past_end_raises_eof_error(self, tmp_path):
        ch = self._open_channel(tmp_path)
        # wavetrack vide → n'importe quelle position est au-delà de la fin
        with pytest.raises(EOFError):
            ch.seek(100)
