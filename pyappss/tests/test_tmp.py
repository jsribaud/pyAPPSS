import pytest

from pyappss import tmp

def test_tmp():
    tmp.tmp()

def test_tmp2():
    val = tmp.tmp2()

    assert val == 2
