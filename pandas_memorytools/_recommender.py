import pandas as pd

from pandas_memorytools._memory_usage import memory_usage
from pandas_memorytools._shrink_dtypes import shrink_dtype


def recommend_dtype(series: pd.Series, **kwargs) -> pd.DataFrame:
    series_shrunk = shrink_dtype(series, **kwargs)
    original_bytes = memory_usage(series)
    new_bytes = memory_usage(series_shrunk)
    return pd.DataFrame(
        {
            "column": [series.name],
            "original_dtype": [series.dtype],
            "recommended_dtype": [series_shrunk.dtype],
            "new_percent": [round(new_bytes / original_bytes, 3)],
            "original_bytes": [original_bytes],
            "new_bytes": [new_bytes],
        }
    ).set_index("column")


def recommend_dtypes(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return pd.concat(
        r for _, col in df.items() if (r := recommend_dtype(col, **kwargs)) is not None
    )
