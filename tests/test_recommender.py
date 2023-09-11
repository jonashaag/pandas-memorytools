import pandas as pd

from pandas_memorytools import recommend_dtypes


def test_recommend_dtypes():
    pd.testing.assert_frame_equal(
        recommend_dtypes(
            pd.DataFrame(
                {
                    "int": [42, 42],
                    "str": ["a", "bc"],
                }
            )
        ),
        pd.DataFrame(
            {
                "column": ["int", "str"],
                "original_dtype": ["int64", "object"],
                "recommended_dtype": ["uint8", "string"],
                "new_percent": [0.125, 0.137],
                "original_bytes": [16, 117],
                "new_bytes": [2, 16],
            }
        ).set_index("column"),
    )
