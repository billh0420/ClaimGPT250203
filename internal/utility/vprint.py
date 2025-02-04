# vprint.py

import sys

verbosity = 1

def vprint(vlevel, *args):
    if verbosity >= vlevel:
        print(*args, file=sys.stderr)
