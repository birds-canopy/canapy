# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
"""Tests for canapy/utils/tempstorage.py — tempfile and close_tempfiles."""

import pytest

from canapy.utils import tempstorage
from canapy.utils.tempstorage import tempfile, close_tempfiles


class TestTempfile:
    def setup_method(self):
        close_tempfiles()

    def teardown_method(self):
        close_tempfiles()

    def test_returns_file_like_object(self):
        fp = tempfile()
        assert hasattr(fp, "read")
        assert hasattr(fp, "write")

    def test_file_is_writable(self):
        fp = tempfile()
        fp.write(b"hello")
        fp.seek(0)
        assert fp.read() == b"hello"

    def test_registers_one_file(self):
        tempfile()
        assert len(tempstorage._registry) == 1

    def test_registers_multiple_files(self):
        tempfile()
        tempfile()
        assert len(tempstorage._registry) == 2

    def test_close_tempfiles_clears_registry(self):
        tempfile()
        tempfile()
        close_tempfiles()
        assert len(tempstorage._registry) == 0

    def test_close_tempfiles_idempotent(self):
        close_tempfiles()
        close_tempfiles()
        assert len(tempstorage._registry) == 0
