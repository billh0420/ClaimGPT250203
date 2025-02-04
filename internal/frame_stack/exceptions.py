# exceptions.py

class MMError(Exception):
    pass


class MMKeyError(MMError, KeyError):
    pass
