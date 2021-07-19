def is_iterable(some_var):
    try:
        _ = (e for e in some_var)
        return True
    except TypeError:
        return False


def merge_dicts(a, b):
    # Just wrapping this fancy-looking expression (Python 3.5+ only). https://stackoverflow.com/a/26853961
    return {**a, **b}


class asobject(object):
    def __init__(self, d:dict):
        self.__dict__ = dict(d.items())


