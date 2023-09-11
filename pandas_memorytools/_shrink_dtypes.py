import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype
from pandas.core.dtypes.base import ExtensionDtype


def shrink_dtype(series: pd.Series, **kwargs) -> pd.Series:
    for dtype in _get_possible_dtypes(series, **kwargs):
        try:
            return _safe_cast(series, dtype)
        except (TypeError, pd.errors.IntCastingNaNError, OverflowError):
            pass
    return series


def shrink_dtypes(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    new_series = {col: shrink_dtype(df[col], **kwargs) for col in list(df.columns)}
    return df.assign(**new_series)


def _get_possible_dtypes(
    series,
    *,
    enable_extension_types: bool = True,
    # enable_float16: bool = False,
    # enable_float32: bool = False,
) -> list[type[np.generic]] | list[ExtensionDtype]:
    if len(series) == 0 or pd.isna(series).all():
        return []
    if is_numeric_dtype(series):
        if series.hasnans:
            if enable_extension_types:
                if series.min() < 0:
                    return [pd.Int8Dtype(), pd.Int16Dtype(), pd.Int32Dtype()]
                else:
                    return [pd.UInt8Dtype(), pd.UInt16Dtype(), pd.UInt32Dtype()]
            else:
                return []
        else:
            if series.min() < 0:
                return [np.int8, np.int16, np.int32]
            else:
                return [np.uint8, np.uint16, np.uint32]
    elif enable_extension_types and is_object_dtype(series):
        return [pd.StringDtype()]
    else:
        return []


def _safe_cast(
    series: pd.Series, numpy_dtype: type[np.generic] | ExtensionDtype
) -> pd.Series:
    cast_series = series.astype(numpy_dtype)
    diffs = cast_series != series
    if diffs.any():
        raise TypeError(
            f"Couldn't safely to {numpy_dtype}. Incompatible values: {series[diffs]}"
        )
    return cast_series
