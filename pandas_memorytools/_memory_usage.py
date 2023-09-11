import sys

import pandas as pd
from pandas.api.types import is_object_dtype

_GETSIZEOF_TYPES = (str, bytes)
_COLLECTION_TYPES = (list, tuple, set)
_MAPPING_TYPES = (dict,)


def memory_usage(
    obj: pd.DataFrame | pd.Series, seen_objs: set | None = None
) -> int | pd.Series:
    if seen_objs is None:
        seen_objs = get_base_seen_objs()

    if isinstance(obj, pd.DataFrame):
        return obj.apply(lambda series: memory_usage(series, seen_objs), axis=0)

    if not is_object_dtype(obj):
        return obj.nbytes

    # TODO 32 bit
    return sum(8 + get_obj_size(item, seen_objs) for item in obj.to_numpy())


def get_obj_size(obj, seen_objs) -> int:
    n_seen = len(seen_objs)
    seen_objs.add(id(obj))
    first_seen = len(seen_objs) != n_seen
    if not first_seen:
        return 0
    if isinstance(obj, _GETSIZEOF_TYPES):
        return sys.getsizeof(obj)
    if isinstance(obj, _COLLECTION_TYPES):
        return sys.getsizeof(obj) + sum(get_obj_size(item, seen_objs) for item in obj)
    if isinstance(obj, _MAPPING_TYPES):
        return (
            sys.getsizeof(obj)
            + sum(get_obj_size(item, seen_objs) for item in obj.keys())
            + sum(get_obj_size(item, seen_objs) for item in obj.items())
        )
    raise NotImplementedError(type(obj))


def get_base_seen_objs():
    return set(map(id, {None, True, False, ""}))
