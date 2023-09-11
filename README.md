# pandas-memorytools

A collection of tools to inspect and optimize Pandas memory usage.

## Dtype recommender

pandas-memorytools can recommend smaller dtypes to reduce memory consumption of your dataframe with no loss of data.

For example, these are the recommendations for the [NYC Yello Taxi Trip Records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page):

```py
>>> from pandas_memorytools import recommend_dtypes
>>> df_sample = df.sample(frac=0.01)
>>> recommend_dtypes(df_sample).sort_values("new_percent")
```

| column                | original_dtype   | recommended_dtype   |   new_percent |   original_bytes |   new_bytes |
|:----------------------|:-----------------|:--------------------|--------------:|-----------------:|------------:|
| VendorID              | int64            | uint8               |         0.125 |          2453416 |      306677 |
| payment_type          | int64            | uint8               |         0.125 |          2453416 |      306677 |
| passenger_count       | float64          | UInt8               |         0.25  |          2453416 |      613354 |
| RatecodeID            | float64          | UInt8               |         0.25  |          2453416 |      613354 |
| PULocationID          | int64            | uint16              |         0.25  |          2453416 |      613354 |
| DOLocationID          | int64            | uint16              |         0.25  |          2453416 |      613354 |
| tpep_dropoff_datetime | datetime64[us]   | datetime64[us]      |         1     |          2453416 |     2453416 |
| ...                   | ...              | ...                 |         ...   |              ... |         ... |

## Memory consumption

pandas-memorytools features a more accurate effective memory consumption reporting than the following alternatives:

- Pandas' `.memory_usage()`: Systematically overestimates the memory consumption of reused Python objects like small strings.
  Does not support reporting memory usage of non-standard objects in columns, like lists.
- `sys.getsizeof()`: Uses `.memory_usage()` under the hood, so has the same limitations as `.memory_usage()`.
- Pympler: Similarly, systematically overestimates the memory consumption of reused Python objects like small strings.

Note: Like Pandas' `.memory_usage()`, pandas-memorytools does not take into account the size of the `pd.DataFrame` and `pd.Series` objects (or any other non-data objects).
It assumes a size of 0 bytes for these objects. This is a simplifcation that is safe to make because for all relevant data sizes the size of these objects is negigible.


### Comparison to Pandas

Here is an example from the NYC Taxi dataset that shows the difference to Pandas' reporting:


```py
>>> from pandas_memorytools import memory_usage
>>> df_sample["store_and_fwd_flag"].memory_usage(deep=True)
1999098
>>> memory_usage(df_sample["store_and_fwd_flag"])
245444
```

Pandas' memory reporting is off by a factor of 8 because all Python strings in the column are cached:


```py
>>> df["store_and_fwd_flag"].unique()
array(['N', 'Y', None], dtype=object)
# Small strings and constants like 'None' are always cached by Python.
```

If your string columns have a lot of Nones, the Pandas result will be off by a factor of 3:

```py
>>> lots_of_nones = pd.Series([None] * 1_000_000)
>>> lots_of_nones.memory_usage(deep=True)
24000132
>>> memory_usage(lots_of_nones)
8000000
```
