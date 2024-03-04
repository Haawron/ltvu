import numpy as np

from ltvu.objects import *



def get_near_gt_interval(
    itvl_full: IntervalType,
    itvl_gt: IntervalType,
    l_window_sec: float = 30.,
    max_window_off_ratio = 2/3,
):
    """
    itvl_full: [s, e] in seconds.
    itvl_gt: [s, e] in seconds.
    l_window_sec: The window size in seconds.
    max_window_offset_ratio: The max ratio w.r.t the window size of the extent that a window can get out of the GT interval.
    """
    itvl_full = Interval(itvl_full)
    itvl_gt = Interval(itvl_gt)
    assert itvl_full.l > l_window_sec
    assert np.isclose(itvl_full.idxs_per_sec, itvl_gt.idxs_per_sec)
    assert itvl_full & itvl_gt is not None
    idxs_per_sec = itvl_full.idxs_per_sec
    l_window = Timestamp(float(l_window_sec), idxs_per_sec)
    ts_min = max(itvl_gt.s - l_window * max_window_off_ratio, itvl_full.s)
    te_max = min(itvl_gt.e + l_window * max_window_off_ratio, itvl_full.e)
    ts_max = te_max - l_window
    ts_sec = np.random.uniform(ts_min, ts_max)
    ts = Timestamp(ts_sec, idxs_per_sec)
    te = ts + l_window
    itvl_new = Interval(ts, te)
    itvl_gt_new = itvl_new & itvl_gt
    return itvl_new, itvl_gt_new
