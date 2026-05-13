# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
"""Tests for canapy/utils/exceptions.py — custom exception classes."""

import pytest

from canapy.utils.exceptions import NotTrainableError, MissingData, NotTrainedError


class TestNotTrainableError:
    def test_is_exception(self):
        assert issubclass(NotTrainableError, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(NotTrainableError):
            raise NotTrainableError("corpus has no training data")

    def test_message_preserved(self):
        msg = "no training sequences available"
        with pytest.raises(NotTrainableError) as exc_info:
            raise NotTrainableError(msg)
        assert msg in str(exc_info.value)


class TestMissingData:
    def test_is_exception(self):
        assert issubclass(MissingData, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(MissingData):
            raise MissingData("syn_mfcc not found")

    def test_message_preserved(self):
        msg = "resource 'syn_mfcc' is missing"
        with pytest.raises(MissingData) as exc_info:
            raise MissingData(msg)
        assert msg in str(exc_info.value)

    def test_not_caught_as_wrong_type(self):
        with pytest.raises(MissingData):
            try:
                raise MissingData("x")
            except NotTrainableError:
                pass


class TestNotTrainedError:
    def test_is_exception(self):
        assert issubclass(NotTrainedError, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(NotTrainedError):
            raise NotTrainedError("annotator not fitted")

    def test_message_preserved(self):
        msg = "call .fit() first"
        with pytest.raises(NotTrainedError) as exc_info:
            raise NotTrainedError(msg)
        assert msg in str(exc_info.value)

    def test_distinct_from_other_errors(self):
        with pytest.raises(NotTrainedError):
            try:
                raise NotTrainedError("x")
            except MissingData:
                pass
