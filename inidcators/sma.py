"""
Simple Moving Average (SMA) - O(1) incremental
"""

from numba import njit
import numpy as np


@njit
def calculate_sma(window, idx, count, sum_val, new_price, period):
    """
    O(1) SMA update

    Args:
        window: circular buffer
        idx: current index
        count: values in buffer
        sum_val: running sum
        new_price: new price
        period: SMA period

    Returns: (sma, new_idx, new_count, new_sum)
    """
    if count == period:
        sum_val = sum_val - window[idx] + new_price
    else:
        sum_val += new_price
        count += 1

    window[idx] = new_price
    idx = (idx + 1) % period

    sma = sum_val / period if count == period else 0.0

    return sma, idx, count, sum_val