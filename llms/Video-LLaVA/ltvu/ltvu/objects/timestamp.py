from typing import Union
from functools import total_ordering

import numpy as np


TimestampType = Union['Timestamp', int, float]

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
            sec_or_idx = sec_or_idx.copy()
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

    def copy(self):
        return Timestamp(self.sec, self.idxs_per_sec)

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
    def __truediv__(self, other: TimestampType):
        if isinstance(other, int|float):
            return Timestamp(self.sec / other, self.idxs_per_sec)
        elif isinstance(other, Timestamp):
            raise ValueError(f'Not allowed to divide two Timestamps.')
    def __rtruediv__(self, other: TimestampType):
        return self.__truediv__(other)
    def __itruediv__(self, other: TimestampType):
        if isinstance(other, int|float):
            self.sec /= other
        elif isinstance(other, Timestamp):
            raise ValueError(f'Not allowed to divide two Timestamps.')
        return self

    def __array__(self, dtype=None):
        if dtype is None or np.issubdtype(dtype, np.floating):
            return np.asarray(self.sec)
        if dtype is not None and np.issubdtype(dtype, np.integer):
            return np.asarray(self.idx)
        raise ValueError(f'Unsupported dtype: {dtype}')

    def _as_timestamp_if_valid(self, other: TimestampType):
        if isinstance(other, (int, float)):
            other = Timestamp(other, self.idxs_per_sec)
        elif isinstance(other, Timestamp):
            self._check_rate(other)
        return other
    def _check_rate(self, other: 'Timestamp'):
        if not np.isclose(self.idxs_per_sec, other.idxs_per_sec):
            raise ValueError(f'Inconsistent fps: {self.idxs_per_sec} vs {other.idxs_per_sec}')
