from typing import Union, Sequence
from functools import total_ordering

import numpy as np


TimestampType = Union['Timestamp', int, float]
IntervalType = Union['Interval', Sequence[TimestampType], np.ndarray[int|float]]


def ensure_timestamp_like(t):
    if not isinstance(t, (int, float, Timestamp)):
        raise TypeError(f'Unsupported type for Timestamp: {type(t).__name__}')


@total_ordering  # defines other comparison methods, may be slow
class Timestamp:
    def __init__(self,
        sec_or_idx: TimestampType,
        idxs_per_sec: float = 30,  # [ FPS ] or [ total frames(or features) / total seconds ]
    ):
        ensure_timestamp_like(sec_or_idx)
        self.idxs_per_sec = idxs_per_sec
        if isinstance(sec_or_idx, int):
            self.idx = sec_or_idx
        elif isinstance(sec_or_idx, float):
            self.sec = sec_or_idx
        elif isinstance(sec_or_idx, Timestamp):
            self.idxs_per_sec = sec_or_idx.idxs_per_sec
            self.sec = sec_or_idx.sec

    @property
    def idx(self):
        return self._idx
    @property
    def sec(self):
        return self._sec
    @idx.setter
    def idx(self, idx):
        self._idx = idx
        self._sec = idx / self.idxs_per_sec
    @sec.setter
    def sec(self, sec):
        self._sec = sec
        self._idx = int(round(sec * self.idxs_per_sec, 0))

    def __str__(self):
        return f'[{self.idx} = {self.sec:.2f}s * {self.idxs_per_sec:.2f} idxs/s]'
    def __repr__(self):
        return f'Timestamp({self.sec:.2f}s, {self.idxs_per_sec:.2f})'
    def __index__(self):
        return self.idx
    def __float__(self):
        return self.sec
    def __int__(self):
        return self.idx

    def __eq__(self, other: TimestampType):
        other = self._as_timestamp_if_valid(other)
        return self.idx == other.idx
    def __lt__(self, other: TimestampType):
        other = self._as_timestamp_if_valid(other)
        return self.idx < other.idx

    def __add__(self, other: TimestampType):
        other = self._as_timestamp_if_valid(other)
        return Timestamp(self.sec + other.sec, self.idxs_per_sec)
    def __radd__(self, other: TimestampType):
        return self.__add__(other)
    def __iadd__(self, other: TimestampType):
        other = self._as_timestamp_if_valid(other)
        self.sec += other.sec
        return self
    def __sub__(self, other: TimestampType):
        other = self._as_timestamp_if_valid(other)
        return Timestamp(self.sec - other.sec, self.idxs_per_sec)
    def __rsub__(self, other: TimestampType):
        return self.__sub__(other)
    def __isub__(self, other: TimestampType):
        other = self._as_timestamp_if_valid(other)
        self.sec -= other.sec
        return self

    def __mul__(self, other: TimestampType):
        if isinstance(other, int|float):
            return Timestamp(self.sec * other, self.idxs_per_sec)
        elif isinstance(other, Timestamp):
            raise ValueError(f'Not allowed to multiply two Timestamps.')
    def __rmul__(self, other: TimestampType):
        return self.__mul__(other)
    def __imul__(self, other: TimestampType):
        if isinstance(other, int|float):
            self.sec *= other
        elif isinstance(other, Timestamp):
            raise ValueError(f'Not allowed to multiply two Timestamps.')
        return self

    def __array__(self, dtype=None):
        if dtype is None or np.issubdtype(dtype, np.floating):
            return np.asarray(self.sec)
        if dtype is not None and np.issubdtype(dtype, np.integer):
            return np.asarray(self.idx)
        raise ValueError(f'Unsupported dtype: {dtype}')

    def _as_timestamp_if_valid(self, other: TimestampType):
        other = Timestamp(other)
        self._check_fps(other)
        return other
    def _check_fps(self, other: 'Timestamp'):
        if not np.isclose(self.idxs_per_sec, other.idxs_per_sec):
            raise ValueError(f'Inconsistent fps: {self.idxs_per_sec} vs {other.idxs_per_sec}')


class Interval:
    def __init__(self,
        s: TimestampType|IntervalType,
        e: TimestampType|None = None
    ):
        if e is None:
            if isinstance(s, Sequence):
                s, e = s
            elif isinstance(s, Interval):
                s, e = s.s, s.e
        self.s = Timestamp(s)
        self.e = Timestamp(e)
        if self.s > self.e:
            raise ValueError('The start time should be less than the end time.')

    def iou(self, others: IntervalType|Sequence[IntervalType]):
        """intersection over union"""
        others = np.array(others, dtype=float)
        unpack = False
        if others.ndim == 1:
            others = others.reshape(1, -1)
            unpack = True
        elif others.ndim != 2:
            raise ValueError(f'Unsupported shape for others: {others.shape}')
        s = np.maximum(self.s.sec, others[:, 0])
        e = np.minimum(self.e.sec, others[:, 1])
        i = e - s
        u = np.maximum(self.l.sec + (others[:, 1] - others[:, 0]) - i)
        if unpack:
            return (i / u).item()
        return i / u

    @property
    def l(self):
        return self.e - self.s

    def __str__(self) -> str:
        return f'[{self.s.sec:.2f}s, {self.e.sec:.2f}s]'
    def __repr__(self) -> str:
        return f'Interval({repr(self.s)}, {repr(self.e)})'

    def __matmul__(self, others: Sequence[IntervalType]):  # @ operator
        return self.iou(others)
    def __imatmul__(self, others: Sequence[IntervalType]):  # @= operator
        return self.iou(others)
    def __rmatmul__(self, others: Sequence[IntervalType]):  # @ operator
        return self.iou(others)

    def __and__(self, other: IntervalType):
        """intersection"""
        if isinstance(other, list):
            other = Interval(other)
        if self.s > other.e or self.e < other.s:
            return 0
        return Interval(max(self.s, other.s), min(self.e, other.e))
    def __or__(self, other: IntervalType):
        """union"""
        if isinstance(other, list):
            other = Interval(other[0], other[1])
        if self.s.sec > other.e.sec or self.e.sec < other.s.sec:
            return 0
        return Interval(min(self.s.sec, other.s.sec), max(self.e.sec, other.e.sec))

    def __array__(self, dtype=None):
        return np.array([self.s, self.e], dtype=dtype)
    def __len__(self):
        return 2
    def __getitem__(self, idx):
        if idx not in [0, 1]:
            raise IndexError(f'Index out of range: {idx}')
        return [self.s, self.e][idx]

    def _as_interval_if_valid(self, other: IntervalType):
        other = Interval(other)
        self._check_fps(other)
        return other
    def _check_fps(self, other: 'Interval'):
        if not np.isclose(self.s.idxs_per_sec, other.s.idxs_per_sec):
            raise ValueError(f'Inconsistent fps: {self.s.idxs_per_sec} vs {other.s.idxs_per_sec}')


if __name__ == '__main__':
    from rich.console import Console
    from rich.syntax import Syntax


    def print_highlighted_expr_and_output(exprs):
        console = Console()
        results = []
        for expr in exprs.strip().split('\n'):
            expr = expr.strip()
            if expr.startswith('#'):
                continue
            result = f'>>> {expr}\n{eval(expr)}'
            results.append(result)
        results = '\n\n'.join(results)
        syntax = Syntax(results, "python", theme="gruvbox-dark", line_numbers=False)
        console.print(syntax)
        print()


    t1 = Timestamp(360. - 1e-7, 30)  # 360 seconds in 30 fps
    exprs = """\
    t1
    t1.idx
    np.arange(14400)[t1]
    np.array(t1)
    np.array([t1]*10)
    np.array([t1]*10, dtype=int)
    Timestamp(t1)"""
    print_highlighted_expr_and_output(exprs)

    t2 = Timestamp(450, 14400/480)  # 450th frame in 480 seconds
    exprs = """\
    t2
    t2.idx
    t2.sec
    np.arange(14400)[t2]
    t2 + 1
    1 + t2
    t2 * 2
    t2 * 2.
    t2 == 450
    """
    print_highlighted_expr_and_output(exprs)

    num_features = 897
    duration_sec = 480
    features_per_second = num_features / duration_sec

    s, e = Timestamp(450, features_per_second), Timestamp(600, features_per_second)
    intval0 = Interval(s, e)
    intval1 = [Timestamp(500, features_per_second), Timestamp(715, features_per_second)]
    intval1 = Interval(intval1)
    exprs = """\
    intval0
    intval1
    intval0.s, intval0.e
    intval0 & intval1
    intval0 | intval1
    intval0.iou(intval1)
    intval0 @ intval1
    intval0 @ [intval1, [500, 600]]
    np.array(intval0)
    np.array([Interval(i, i+1) for i in range(10)])
    """
    print_highlighted_expr_and_output(exprs)
