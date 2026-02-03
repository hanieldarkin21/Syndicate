"""
MACD - O(1) incremental
Trend-following momentum indicator
"""

from numba import njit


@njit
def calculate_macd(prev_fast_ema, prev_slow_ema, prev_signal_ema, new_price,
                   fast_period, slow_period, signal_period):
    """
    O(1) MACD update

    MACD Line = Fast EMA - Slow EMA
    Signal Line = EMA of MACD Line
    Histogram = MACD Line - Signal Line

    Args:
        prev_fast_ema: previous fast EMA (typically 12)
        prev_slow_ema: previous slow EMA (typically 26)
        prev_signal_ema: previous signal line EMA (typically 9)
        new_price: new price
        fast_period: fast EMA period
        slow_period: slow EMA period
        signal_period: signal EMA period

    Returns: (macd_line, signal_line, histogram, new_fast_ema, new_slow_ema, new_signal_ema)
    """
    # Calculate alphas
    fast_alpha = 2.0 / (fast_period + 1)
    slow_alpha = 2.0 / (slow_period + 1)
    signal_alpha = 2.0 / (signal_period + 1)

    # Update fast EMA
    if prev_fast_ema == 0.0:
        fast_ema = new_price
    else:
        fast_ema = fast_alpha * new_price + (1.0 - fast_alpha) * prev_fast_ema

    # Update slow EMA
    if prev_slow_ema == 0.0:
        slow_ema = new_price
    else:
        slow_ema = slow_alpha * new_price + (1.0 - slow_alpha) * prev_slow_ema

    # Calculate MACD line
    macd_line = fast_ema - slow_ema

    # Update signal line
    if prev_signal_ema == 0.0:
        signal_line = macd_line
    else:
        signal_line = signal_alpha * macd_line + (1.0 - signal_alpha) * prev_signal_ema

    # Calculate histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram, fast_ema, slow_ema, signal_line