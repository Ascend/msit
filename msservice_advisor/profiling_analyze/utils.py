from collections import namedtuple

TARGETS = namedtuple("TARGETS", ["FirstTokenTime", "Throughput"])("FirstTokenTime", "Throughput")

def str_ignore_case(value):
    return value.lower().replace("_", "").replace("-", "")


def walk_dict(data, parent_key=""):
    if isinstance(data, dict):
        for key, value in data.items():
            if not isinstance(value, (dict, tuple, list)):
                yield key, value, parent_key
            else:
                new_key = f"{parent_key}.{key}" if parent_key else key
                yield from walk_dict(value, new_key)
    elif isinstance(data, (tuple, list)):
        for index, item in enumerate(data):
            if not isinstance(item, (dict, tuple, list)):
                yield key, item, parent_key
            else:
                new_key = f"{parent_key}.{index}" if parent_key else index
                yield from walk_dict(item, new_key)
