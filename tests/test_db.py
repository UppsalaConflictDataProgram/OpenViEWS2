""" Tests for views.utils.db

@TODO: Add a testing db...
"""
import pytest  # type: ignore
from views.utils import db


def test_unpack_fqtable() -> None:
    """ Test unpack fqtable """
    assert db._unpack_fqtable("schema.table") == ("schema", "table")
