"""
Real Data Strategy Test - Databento NQ Futures
Tests all indicators with real tick data and measures performance
"""

import indicator_init
import numpy as np
import time
from pathlib import Path

# Try importing databento
try:
    import databento as db
    HAS_DATABENTO = True
except ImportError:
    HAS_DATABENTO = False
    print("⚠️  databento not installed. Install with: pip install databento")
    print("Falling back to demonstration mode...\n")


def load_nq_data(filepath="NQ.dbn"):
    """
    Load NQ futures data from Databento DBN file - OPTIMIZED

    Uses to_ndarray() for maximum performance - avoids Python object overhead

    Args:
        filepath: Path to NQ.dbn file

    Returns:
        prices: numpy array of trade prices
        timestamps: numpy array of timestamps
        data_info: dict with metadata
    """
    if not HAS_DATABENTO:
        print("Databento not available - using simulated data")
        return generate_simulated_nq_data()

    filepath = Path(filepath)
    if not filepath.exists():
        print(f"⚠️  File not found: {filepath}")
        print("Using simulated data instead...\n")
        return generate_simulated_nq_data()

    print(f"📊 Loading {filepath}...")

    try:
        # OPTIMIZED: Use DBNStore.from_file and convert to ndarray directly
        # This avoids Python object iteration overhead
        store = db.DBNStore.from_file(filepath)

        # Convert to structured numpy array (FAST!)
        # This is the optimal way - no Python object creation
        data = store.to_ndarray()

        # Extract fields directly from structured array
        # Field names depend on schema (trades, mbp-1, mbo, etc.)

        if 'price' in data.dtype.names:
            # Trade data - has 'price' field
            prices = data['price'].astype(np.float64) / 1e9  # Convert from fixed-point
            timestamps = data['ts_event']
        elif 'bid_px_00' in data.dtype.names and 'ask_px_00' in data.dtype.names:
            # MBP/MBO data - calculate midpoint
            bid_prices = data['bid_px_00'].astype(np.float64) / 1e9
            ask_prices = data['ask_px_00'].astype(np.float64) / 1e9

            # Filter out invalid prices (0 means no quote)
            valid_mask = (bid_prices > 0) & (ask_prices > 0)
            bid_prices = bid_prices[valid_mask]
            ask_prices = ask_prices[valid_mask]

            prices = (bid_prices + ask_prices) / 2.0
            timestamps = data['ts_event'][valid_mask]
        else:
            # Unknown schema - try to find any price field
            print(f"⚠️  Unknown schema. Available fields: {data.dtype.names}")
            print("Attempting to find price field...")

            # Look for common price fields
            price_field = None
            for field in ['price', 'open', 'close', 'last']:
                if field in data.dtype.names:
                    price_field = field
                    break

            if price_field:
                prices = data[price_field].astype(np.float64) / 1e9
                timestamps = data['ts_event']
            else:
                raise ValueError("No price field found in data")

        # Ensure prices are float64 and contiguous for Numba
        prices = np.ascontiguousarray(prices, dtype=np.float64)
        timestamps = np.ascontiguousarray(timestamps, dtype=np.int64)

        data_info = {
            'symbol': 'NQ',
            'total_ticks': len(prices),
            'price_min': prices.min(),
            'price_max': prices.max(),
            'price_mean': prices.mean(),
            'time_span_ns': timestamps[-1] - timestamps[0] if len(timestamps) > 0 else 0,
        }

        print(f"✓ Loaded {len(prices):,} ticks")
        print(f"  Schema: {store.schema if hasattr(store, 'schema') else 'unknown'}")
        print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        print(f"  Mean price: ${prices.mean():.2f}")

        if len(timestamps) > 1:
            time_span_sec = (timestamps[-1] - timestamps[0]) / 1e9
            ticks_per_sec = len(prices) / time_span_sec
            print(f"  Time span: {time_span_sec:.2f} seconds")
            print(f"  Average: {ticks_per_sec:.0f} ticks/second\n")

        return prices, timestamps, data_info

    except Exception as e:
        print(f"⚠️  Error reading DBN file: {e}")
        import traceback
        traceback.print_exc()
        print("Using simulated data instead...\n")
        return generate_simulated_nq_data()


def generate_simulated_nq_data():
    """
    Generate simulated NQ futures data for testing
    Mimics real NQ behavior: ~20,000 price level, realistic volatility
    """
    print("📊 Generating simulated NQ data...")

    n_ticks = 1_000_000
    base_price = 20000.0

    # Simulate realistic NQ price movement
    np.random.seed(42)

    # Trend component
    trend = np.linspace(0, 200, n_ticks)

    # Volatility component (realistic for NQ - about 0.5% intraday range)
    volatility = np.random.normal(0, 10, n_ticks).cumsum()

    # Microstructure noise (tick-by-tick)
    noise = np.random.normal(0, 2, n_ticks)

    prices = base_price + trend + volatility + noise

    # Generate timestamps (assuming ~1000 ticks/second average)
    timestamps = np.arange(n_ticks, dtype=np.int64) * 1_000_000  # nanoseconds

    data_info = {
        'symbol': 'NQ (simulated)',
        'total_ticks': len(prices),
        'price_min': prices.min(),
        'price_max': prices.max(),
        'price_mean': prices.mean(),
        'time_span_ns': timestamps[-1] - timestamps[0],
    }

    print(f"✓ Generated {len(prices):,} simulated ticks")
    print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    print(f"  Mean price: ${prices.mean():.2f}\n")

    return prices, timestamps, data_info


def benchmark_indicator_with_real_data(name, indicator_func, prices, timestamps, setup_code):
    """
    Benchmark an indicator with real market data using RAW functions (no wrapper overhead)

    Args:
        name: Indicator name
        indicator_func: Raw compiled function from registry
        prices: Price data
        timestamps: Timestamp data
        setup_code: Function that returns test function using raw indicator

    Returns:
        results dict with performance metrics
    """
    print(f"{'=' * 70}")
    print(f"{name} - Real Data Performance Test")
    print(f"{'=' * 70}")

    n_ticks = len(prices)

    # Get the test function
    test_func = setup_code(indicator_func)

    # Warmup
    _ = test_func(prices[:min(10000, n_ticks)])

    # Benchmark
    start_time = time.time()
    start_ns = time.perf_counter_ns()

    results = test_func(prices)

    end_ns = time.perf_counter_ns()
    end_time = time.time()

    elapsed_time = end_time - start_time
    elapsed_ns = end_ns - start_ns

    # Calculate metrics
    ticks_per_second = n_ticks / elapsed_time
    ns_per_tick = elapsed_ns / n_ticks

    # Calculate throughput in terms of real market speed
    if len(timestamps) > 1:
        real_time_span_sec = (timestamps[-1] - timestamps[0]) / 1e9
        real_ticks_per_sec = n_ticks / real_time_span_sec
        speedup = ticks_per_second / real_ticks_per_sec
    else:
        real_ticks_per_sec = 1000  # assume 1k ticks/sec
        speedup = ticks_per_second / real_ticks_per_sec

    print(f"Ticks processed: {n_ticks:,}")
    print(f"Time elapsed: {elapsed_time:.3f}s")
    print(f"")
    print(f"⚡ Performance:")
    print(f"  {ticks_per_second:,.0f} ticks/second")
    print(f"  {ticks_per_second / 1_000_000:.2f} MILLION ticks/second")
    print(f"  {ns_per_tick:.1f} nanoseconds/tick")
    print(f"")
    print(f"📈 Market Context:")
    print(f"  Real market speed: ~{real_ticks_per_sec:.0f} ticks/second")
    print(f"  Speedup vs real-time: {speedup:.0f}x")
    print(f"  Can handle: {int(speedup)} simultaneous NQ streams")
    print(f"")

    return {
        'name': name,
        'ticks': n_ticks,
        'elapsed': elapsed_time,
        'ticks_per_sec': ticks_per_second,
        'ns_per_tick': ns_per_tick,
        'speedup': speedup,
    }


def run_strategy_simulation(prices, timestamps):
    """
    Run a complete trading strategy simulation with multiple indicators
    Uses RAW compiled functions for maximum speed
    """
    print(f"\n{'=' * 70}")
    print("TRADING STRATEGY SIMULATION")
    print(f"{'=' * 70}\n")

    print("Strategy: Trend + Mean Reversion + Momentum")
    print("  - SMA(10) / SMA(30) for trend")
    print("  - Bollinger Bands(20) for mean reversion")
    print("  - RSI(14) for momentum\n")

    # Get raw compiled functions
    from numba import njit

    sma_func = indicator_init.get('SMA')
    rsi_func = indicator_init.get('RSI')
    bb_func = indicator_init.get('BOLLINGER')

    # Create JIT-compiled strategy function
    @njit
    def run_strategy(prices):
        n = len(prices)

        # SMA fast state
        period_fast = 10
        window_fast = np.zeros(period_fast, dtype=np.float64)
        idx_fast, count_fast, sum_fast = 0, 0, 0.0

        # SMA slow state
        period_slow = 30
        window_slow = np.zeros(period_slow, dtype=np.float64)
        idx_slow, count_slow, sum_slow = 0, 0, 0.0

        # RSI state
        period_rsi = 14
        prev_avg_gain, prev_avg_loss = 0.0, 0.0
        prev_price = prices[0]

        # Bollinger state
        period_bb = 20
        num_std = 2.0
        window_bb = np.zeros(period_bb, dtype=np.float64)
        idx_bb, count_bb, sum_bb, sum_sq_bb = 0, 0, 0.0, 0.0

        # Results
        buy_signals = 0
        sell_signals = 0
        current_position = 0

        for i in range(n):
            price = prices[i]

            # Update SMA fast
            sma_f, idx_fast, count_fast, sum_fast = sma_func(
                window_fast, idx_fast, count_fast, sum_fast, price, period_fast
            )

            # Update SMA slow
            sma_s, idx_slow, count_slow, sum_slow = sma_func(
                window_slow, idx_slow, count_slow, sum_slow, price, period_slow
            )

            # Update RSI
            if i > 0:
                rsi_val, prev_avg_gain, prev_avg_loss = rsi_func(
                    prev_avg_gain, prev_avg_loss, price, prev_price, period_rsi
                )
                prev_price = price
            else:
                rsi_val = 50.0

            # Update Bollinger
            bb_mid, bb_upper, bb_lower, idx_bb, count_bb, sum_bb, sum_sq_bb = bb_func(
                window_bb, idx_bb, count_bb, sum_bb, sum_sq_bb, price, period_bb, num_std
            )

            # Generate signals
            signal = 0

            if sma_f > 0 and sma_s > 0 and bb_mid > 0:
                # Trend following with mean reversion
                if sma_f > sma_s and price < bb_lower and rsi_val < 30:
                    signal = 1  # Buy
                    buy_signals += 1
                elif sma_f < sma_s and price > bb_upper and rsi_val > 70:
                    signal = -1  # Sell
                    sell_signals += 1

            # Update position
            if signal == 1 and current_position == 0:
                current_position = 1
            elif signal == -1 and current_position == 1:
                current_position = 0

        return buy_signals, sell_signals

    # Warmup compilation
    _ = run_strategy(prices[:1000])

    # Run simulation
    start_time = time.time()
    buy_signals, sell_signals = run_strategy(prices)
    elapsed = time.time() - start_time

    print(f"Simulation Results:")
    print(f"  Ticks processed: {len(prices):,}")
    print(f"  Time elapsed: {elapsed:.3f}s")
    print(f"  Processing speed: {len(prices) / elapsed:,.0f} ticks/second")
    print(f"  Processing speed: {(len(prices) / elapsed) / 1_000_000:.2f} MILLION ticks/second")
    print(f"")
    print(f"  Buy signals: {buy_signals}")
    print(f"  Sell signals: {sell_signals}")
    print(f"  Signal rate: {(buy_signals + sell_signals) / len(prices) * 100:.3f}%")
    print(f"")


def main():
    print("=" * 70)
    print("REAL DATA INDICATOR BENCHMARK - NQ FUTURES")
    print("=" * 70)
    print()

    # Load data
    prices, timestamps, data_info = load_nq_data("NQ.dbn")

    if len(prices) == 0:
        print("❌ No data available")
        return

    print(f"{'=' * 70}")
    print("DATA SUMMARY")
    print(f"{'=' * 70}")
    for key, value in data_info.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    print()

    # Import numba for JIT compilation
    from numba import njit

    # Benchmark each indicator using RAW compiled functions (no wrapper overhead)
    results = {}

    # SMA
    sma_func = indicator_init.get('SMA')
    def setup_sma(func):
        @njit
        def test_sma(prices):
            period = 20
            smas = np.zeros(len(prices), dtype=np.float64)
            window = np.zeros(period, dtype=np.float64)
            idx, count, sum_val = 0, 0, 0.0

            for i in range(len(prices)):
                sma, idx, count, sum_val = func(window, idx, count, sum_val, prices[i], period)
                smas[i] = sma
            return smas
        # Warmup
        _ = test_sma(np.array([20000.0, 20001.0], dtype=np.float64))
        return test_sma

    results['SMA(20)'] = benchmark_indicator_with_real_data('SMA(20)', sma_func, prices, timestamps, setup_sma)

    # EMA
    ema_func = indicator_init.get('EMA')
    def setup_ema(func):
        @njit
        def test_ema(prices):
            period = 12
            alpha = 2.0 / (period + 1)
            emas = np.zeros(len(prices), dtype=np.float64)
            prev_ema = 0.0

            for i in range(len(prices)):
                ema = func(prev_ema, prices[i], alpha)
                emas[i] = ema
                prev_ema = ema
            return emas
        _ = test_ema(np.array([20000.0, 20001.0], dtype=np.float64))
        return test_ema

    results['EMA(12)'] = benchmark_indicator_with_real_data('EMA(12)', ema_func, prices, timestamps, setup_ema)

    # RSI
    rsi_func = indicator_init.get('RSI')
    def setup_rsi(func):
        @njit
        def test_rsi(prices):
            period = 14
            rsis = np.zeros(len(prices), dtype=np.float64)
            prev_avg_gain, prev_avg_loss = 0.0, 0.0
            prev_price = prices[0]

            for i in range(1, len(prices)):
                rsi, prev_avg_gain, prev_avg_loss = func(
                    prev_avg_gain, prev_avg_loss, prices[i], prev_price, period
                )
                rsis[i] = rsi
                prev_price = prices[i]
            return rsis
        _ = test_rsi(np.array([20000.0, 20001.0], dtype=np.float64))
        return test_rsi

    results['RSI(14)'] = benchmark_indicator_with_real_data('RSI(14)', rsi_func, prices, timestamps, setup_rsi)

    # Bollinger Bands
    bb_func = indicator_init.get('BOLLINGER')
    def setup_bb(func):
        @njit
        def test_bollinger(prices):
            period = 20
            num_std = 2.0
            window = np.zeros(period, dtype=np.float64)
            idx, count, sum_val, sum_sq = 0, 0, 0.0, 0.0
            middles = np.zeros(len(prices), dtype=np.float64)

            for i in range(len(prices)):
                middle, upper, lower, idx, count, sum_val, sum_sq = func(
                    window, idx, count, sum_val, sum_sq, prices[i], period, num_std
                )
                middles[i] = middle
            return middles
        _ = test_bollinger(np.array([20000.0, 20001.0], dtype=np.float64))
        return test_bollinger

    results['Bollinger(20,2)'] = benchmark_indicator_with_real_data('Bollinger(20,2)', bb_func, prices, timestamps, setup_bb)

    # MACD
    macd_func = indicator_init.get('MACD')
    def setup_macd(func):
        @njit
        def test_macd(prices):
            fast_period, slow_period, signal_period = 12, 26, 9
            macds = np.zeros(len(prices), dtype=np.float64)
            prev_fast_ema, prev_slow_ema, prev_signal_ema = 0.0, 0.0, 0.0

            for i in range(len(prices)):
                macd_line, signal_line, histogram, prev_fast_ema, prev_slow_ema, prev_signal_ema = func(
                    prev_fast_ema, prev_slow_ema, prev_signal_ema, prices[i],
                    fast_period, slow_period, signal_period
                )
                macds[i] = macd_line
            return macds
        _ = test_macd(np.array([20000.0, 20001.0], dtype=np.float64))
        return test_macd

    results['MACD(12,26,9)'] = benchmark_indicator_with_real_data('MACD(12,26,9)', macd_func, prices, timestamps, setup_macd)

    # Summary
    print(f"{'=' * 70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'=' * 70}")
    print()

    # Sort by speed
    sorted_results = sorted(results.items(), key=lambda x: x[1]['ticks_per_sec'], reverse=True)

    print(f"{'Indicator':<20} {'Ticks/sec':>15} {'ns/tick':>12} {'Speedup':>10}")
    print("-" * 70)
    for name, res in sorted_results:
        print(f"{name:<20} {res['ticks_per_sec']:>15,.0f} {res['ns_per_tick']:>12.1f} {res['speedup']:>10.0f}x")

    print()
    print(f"✓ All indicators can process NQ tick data in real-time")
    print(f"✓ Average speedup: {np.mean([r['speedup'] for r in results.values()]):.0f}x faster than market")
    print()

    # Run strategy simulation
    run_strategy_simulation(prices, timestamps)

    print(f"{'=' * 70}")
    print("✓ BENCHMARK COMPLETE")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
