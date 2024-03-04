from typing import Union, Sequence

import numpy as np

from .timestamp import Timestamp, TimestampType


IntervalType = Union['Interval', Sequence[TimestampType], np.ndarray[int|float]]


class Interval:
    def __init__(self,
        s: TimestampType|IntervalType,
        e: TimestampType|None = None,
        idxs_per_sec: float = 30,
    ):
        """
        Possible type pairs for (s, e):
        - (TimestampType, TimestampType)
        - (IntervalType, None)
        """
        if e is None:
            if isinstance(s, Sequence):
                s, e = s
                self.s = Timestamp(s, idxs_per_sec=idxs_per_sec)
                self.e = Timestamp(e, idxs_per_sec=idxs_per_sec)
            elif isinstance(s, Interval):
                s = s.copy()
                s, e = s.s, s.e
                self.s = Timestamp(s)
                self.e = Timestamp(e)
        else:
            if type(s) != type(e):
                raise TypeError(f'Inconsistent types for (s, e): {type(s).__name__} vs {type(e).__name__}')
            self.s = Timestamp(s, idxs_per_sec=idxs_per_sec)
            self.e = Timestamp(e, idxs_per_sec=idxs_per_sec)
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
        i = np.maximum(0, e - s)
        u = self.l.sec + (others[:, 1] - others[:, 0]) - i
        if unpack:
            return (i / u).item()
        return i / u

    @property
    def c(self):
        return (self.s + self.e) / 2
    @property
    def w(self):
        return self.e - self.c
    @c.setter
    def c(self, c):
        prev_w = self.w.copy()
        self.s = c - prev_w
        self.e = c + prev_w
    @w.setter
    def w(self, w):
        prev_c = self.c.copy()
        self.s = prev_c - w
        self.e = prev_c + w

    @property
    def l(self):
        return self.e - self.s
    @property
    def idxs_per_sec(self):
        return self.s.idxs_per_sec
    @property
    def idxs(self):
        return np.array([self.s.idx, self.e.idx], dtype=int)
    @property
    def secs(self):
        return np.array([self.s.sec, self.e.sec], dtype=float)

    def copy(self):
        return Interval(self.s, self.e, idxs_per_sec=self.idxs_per_sec)

    def to_dict(self):
        return {
            'start_sec': self.s.sec,
            'end_sec': self.e.sec,
            's_ind': self.s.idx,  # why not start_idx? --> Ego4D's convention
            'e_ind': self.e.idx,
        }

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

    def __add__(self, other: int|float):
        return Interval(self.s + other, self.e + other)
    def __radd__(self, other: int|float):
        return Interval(self.s + other, self.e + other)
    def __iadd__(self, other: int|float):
        self.s += other
        self.e += other
        return self
    def __sub__(self, other: int|float):
        return Interval(self.s - other, self.e - other)
    def __rsub__(self, other: int|float):
        return Interval(self.s - other, self.e - other)
    def __isub__(self, other: int|float):
        self.s -= other
        self.e -= other
        return self
    def __mul__(self, other: int|float):
        itvl = Interval(self)
        itvl.w *= other
        return itvl
    def __rmul__(self, other: int|float):
        return self.__mul__(other)
    def __imul__(self, other: int|float):
        self.w *= other
        return self
    def __truediv__(self, other: int|float):
        itvl = Interval(self.copy())
        itvl.w /= other
        return itvl
    def __rtruediv__(self, other: int|float):
        return self.__truediv__(other)
    def __itruediv__(self, other: int|float):
        self.w /= other
        return self

    def __and__(self, other: IntervalType):
        """intersection"""
        other = self._as_interval_if_valid(other)
        if self.s > other.e or self.e < other.s:
            return None
        return Interval(max(self.s, other.s), min(self.e, other.e))
    def __iand__(self, other: IntervalType):
        other = self._as_interval_if_valid(other)
        if self.s > other.e or self.e < other.s:
            return None
        self.s = max(self.s, other.s)
        self.e = min(self.e, other.e)
        return self
    def __or__(self, other: IntervalType):
        """union"""
        other = self._as_interval_if_valid(other)
        if self.s > other.e or self.e < other.s:
            return self, other
        return Interval(min(self.s, other.s), max(self.e, other.e))
    def __ior__(self, other: IntervalType):
        other = self._as_interval_if_valid(other)
        if self.s > other.e or self.e < other.s:
            return self, other
        self.s = min(self.s, other.s)
        self.e = max(self.e, other.e)
        return self

    def __array__(self, dtype=None):
        return np.array([self.s, self.e], dtype=dtype)
    def __len__(self):
        return 2
    def __getitem__(self, idx):
        if idx not in [0, 1]:
            raise IndexError(f'Index out of range: {idx}')
        return [self.s, self.e][idx]

    def _as_interval_if_valid(self, other: IntervalType):
        if isinstance(other, (list, tuple, np.ndarray)):
            other = Interval(other, idxs_per_sec = self.idxs_per_sec)
        elif isinstance(other, Interval):
            self._check_rate(other)
        return other
    def _check_rate(self, other: 'Interval'):
        if not np.isclose(self.idxs_per_sec, other.idxs_per_sec):
            raise ValueError(f'Inconsistent fps: {self.idxs_per_sec} vs {other.idxs_per_sec}')
