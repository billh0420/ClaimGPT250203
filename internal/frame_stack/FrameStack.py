# from shared.frame_stack
# FrameStack.py

import itertools
import collections

from internal.frame_stack.Frame import Frame
from internal.frame_stack.exceptions import MMError, MMKeyError
from internal.utility.vprint import vprint

class FrameStack(list):
    def push(self):
        self.append(Frame())

    def add_c(self, tok):
        frame = self[-1]
        if tok in frame.c:
            raise MMError('const already defined in scope')
        if tok in frame.v:
            raise MMError('const already defined as var in scope')
        frame.c.add(tok)

    def add_v(self, tok):
        frame = self[-1]
        if tok in frame.v:
            raise MMError(f'var {tok} already defined in scope')
        if tok in frame.c:
            raise MMError('var already defined as const in scope')
        frame.v.add(tok)

    def add_f(self, var, kind, label):
        if not self.lookup_v(var):
            raise MMError('var in $f not defined: {0}'.format(var))
        if not self.lookup_c(kind):
            raise MMError('const in $f not defined {0}'.format(kind))
        frame = self[-1]
        if var in frame.f_labels.keys():
            raise MMError('var in $f already defined in scope')
        frame.f.append((var, kind))
        frame.f_labels[var] = label
        frame.type_code_by_var[var] = kind

    def add_e(self, stat, label):
        frame = self[-1]
        frame.e.append(stat)
        frame.e_labels[tuple(stat)] = label

    def add_d(self, stat):
        frame = self[-1]
        frame.d.update(((min(x, y), max(x, y)) for x, y in itertools.product(stat, stat) if x != y))

    def lookup_c(self, tok):
        return any((tok in fr.c for fr in reversed(self)))

    def lookup_v(self, tok):
        return any((tok in fr.v for fr in reversed(self)))

    def lookup_f(self, var):
        for frame in reversed(self):
            try:
                return frame.f_labels[var]
            except KeyError:
                pass
        raise MMKeyError(var)

    def lookup_d(self, x, y):
        return any(((min(x, y), max(x, y)) in fr.d for fr in reversed(self)))

    def lookup_e(self, stmt):
        stmt_t = tuple(stmt)
        for frame in reversed(self):
            try:
                return frame.e_labels[stmt_t]
            except KeyError:
                pass
        raise MMKeyError(stmt_t)

    def make_assertion(self, stat):
        e_hyps = [eh for fr in self for eh in fr.e]
        mand_vars = {tok for hyp in itertools.chain(e_hyps, [stat]) for tok in hyp if self.lookup_v(tok)}

        dvs = {(x, y) for fr in self for (x, y) in fr.d.intersection(itertools.product(mand_vars, mand_vars))}

        f_hyps = collections.deque()
        for fr in reversed(self):
            for v, k in reversed(fr.f):
                if v in mand_vars:
                    f_hyps.appendleft((k, v))
                    mand_vars.remove(v)

        vprint(18, 'ma:', (dvs, f_hyps, e_hyps, stat))
        return dvs, f_hyps, e_hyps, stat
