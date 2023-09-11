import hypothesis.strategies as st
import numpy as np
import pandas as pd
from hypothesis import assume, given, settings
from hypothesis.extra import numpy as extra_numpy
from hypothesis.extra import pandas as extra_pandas

from pandas_memorytools import shrink_dtype, shrink_dtypes


@given(
    input_series=extra_pandas.series(elements=st.floats() | st.integers()),
    input_dtype=extra_numpy.integer_dtypes() | extra_numpy.floating_dtypes(),
    enable_extension_types=st.booleans(),
)
@settings(max_examples=1_000)
def test_shrink_dtype(input_series, input_dtype, enable_extension_types):
    try:
        input_series = _safe_cast(input_series, input_dtype)
    except Exception:
        assume(False)
        return

    _shrink_kwargs = dict(enable_extension_types=enable_extension_types)
    _shrink = lambda: shrink_dtype(input_series, **_shrink_kwargs)
    _shrink_df = lambda: shrink_dtypes(input_series.to_frame("col1"), **_shrink_kwargs)[
        "col1"
    ]

    if (
        len(input_series) == 0
        or any(map(_is_non_integer, input_series.dropna().to_numpy()))
        or pd.isna(input_series).all()
    ):
        shrunk_series = _shrink()
        assert shrunk_series is input_series
        return

    expected_width, expected_sign = _get_expected_width_and_sign(input_series)
    if expected_sign is None:
        assert _shrink() is input_series
    else:
        if pd.isna(input_series).any():
            if enable_extension_types:
                expected_dtype = getattr(
                    pd, f"{expected_sign}Int{expected_width}Dtype"
                )()
            else:
                expected_dtype = input_series.dtype
                assert _shrink() is input_series
        else:
            expected_dtype = getattr(np, f"{expected_sign.lower()}int{expected_width}")
        assert _shrink().dtype == expected_dtype
        assert _shrink_df().dtype == expected_dtype
    assert _to_py(input_series) == _to_py(_shrink())


def _to_py(series):
    return [
        None if pd.isna(v) else int(v)
        for v in series.replace(np.inf, np.nan).replace(-np.inf, np.nan).to_numpy()
    ]


def _get_expected_width_and_sign(series):
    series = series.dropna()
    if series.min() < 0:
        for width in [8, 16, 32]:
            try:
                _safe_cast(series, getattr(np, f"int{width}"))
                return width, ""
            except Exception:
                pass
    else:
        for width in [8, 16, 32]:
            try:
                _safe_cast(series, getattr(np, f"uint{width}"))
                return width, "U"
            except Exception:
                pass
    return None, None


def _is_non_integer(x):
    return isinstance(x, float) and not x.is_integer()


def _safe_cast(series, dtype):
    series2 = series.astype(dtype)
    pd.testing.assert_series_equal(series2, series, check_dtype=False, check_exact=True)
    return series2


def test_shrink_dtype_strings():
    series = pd.Series(["a", None])
    assert shrink_dtype(series).dtype == "string"
    assert shrink_dtype(series, enable_extension_types=False).dtype == "object"
