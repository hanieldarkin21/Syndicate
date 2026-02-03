"""
Relative Strength Index (RSI) - O(1) incremental
"""

from numba import njit


@njit
def calculate_rsi(prev_avg_gain, prev_avg_loss, new_price, prev_price, period):
    """
    O(1) RSI update using Wilder's smoothing

    Args:
        prev_avg_gain: previous average gain
        prev_avg_loss: previous average loss
        new_price: new price
        prev_price: previous price
        period: RSI period (typically 14)

    Returns: (rsi, new_avg_gain, new_avg_loss)
    """
    # Calculate price change
    change = new_price - prev_price
    gain = max(change, 0.0)
    loss = max(-change, 0.0)

    # Wilder's smoothing (exponential moving average)
    if prev_avg_gain == 0.0 and prev_avg_loss == 0.0:
        # First calculation
        avg_gain = gain
        avg_loss = loss
    else:
        # Incremental update
        avg_gain = (prev_avg_gain * (period - 1) + gain) / period
        avg_loss = (prev_avg_loss * (period - 1) + loss) / period

    # Calculate RSI
    if avg_loss == 0.0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi, avg_gain, avg_loss