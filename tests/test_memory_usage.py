import random
import sys

import pandas as pd

from pandas_memorytools import memory_usage

_sizeof_list = sys.getsizeof([])
_sizeof_str = sys.getsizeof("")


def test_memory_usage_basics():
    assert memory_usage(pd.Series([])) == 0
    for n in range(1, 100):
        assert memory_usage(pd.Series(range(n))) == 8 * n
        assert memory_usage(pd.Series(range(n)) + 0.1) == 8 * n
    assert memory_usage(pd.Series([""])) == 1 * 8
    assert memory_usage(pd.Series(["", ""])) == 2 * 8
    assert memory_usage(pd.Series(["", "", None, True, False])) == 5 * 8


def test_memory_usage_strings():
    s1 = "".join(random.choice("abc") for _ in range(100))
    s2 = "".join(random.choice("abc") for _ in range(100))
    assert memory_usage(pd.Series([s1])) == _sizeof_str + 1 * 8 + 100
    assert memory_usage(pd.Series([s1, s1])) == _sizeof_str + 2 * 8 + 100
    assert memory_usage(pd.Series([s1, s2])) == 2 * (_sizeof_str + +1 * 8 + 100)


def test_memory_usage_lists():
    assert memory_usage(pd.Series([[]])) == 1 * (8 + _sizeof_list)
    assert memory_usage(pd.Series([[], []])) == 2 * (8 + _sizeof_list)
    assert memory_usage(pd.Series([[], [], None])) == 2 * (8 + _sizeof_list) + 1 * 8
