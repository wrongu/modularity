def is_iterable(some_var):
    try:
        _ = (e for e in some_var)
        return True
    except TypeError:
        return False


class asobject(object):
    def __init__(self, d:dict):
        self.__dict__ = dict(d.items())


