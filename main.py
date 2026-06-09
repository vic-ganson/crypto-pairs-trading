import pandas as pd
import numpy as np
from datetime import datetime

from src.data_fetch import load_prices
from src.pairs_selection import find_best_pairs
from src.signals import gen_signals
from src.portfolio import gen_portfolio, compute_net_returns
from src.performance import (
    compute_stats,
    plot_cumulative_returns,
    plot_drawdown,
    plot_rolling_sharpe
)

# ── Config ───────────────────────────────────────────────────────────────────

COIN_IDS = [
    'bitcoin', 'ethereum', 'binancecoin', 'ripple', 'cardano',
    'solana', 'polkadot', 'dogecoin', 'avalanche-2', 'chainlink',
    'litecoin', 'uniswap', 'cosmos', 'algorand', 'stellar',
    'filecoin', 'tron', 'monero', 'ethereum-classic', 'vechain'
]

IN_SAMPLE_START  = "2018-01-01"
IN_SAMPLE_END    = "2021-12-31"
OUT_SAMPLE_START = "2022-01-01"
OUT_SAMPLE_END   = "2024-08-16"

EXIT_THRESHOLD = 0.2
KALMAN_DELTA   = 1e-4
KALMAN_R       = 1e-2
ZSCORE_WINDOW  = 90
TCOST_BPS      = 20

def main():

    # Loading data
    print("\n[1/5] Loading price data...")
    crypto_px = load_prices(COIN_IDS, IN_SAMPLE_START, OUT_SAMPLE_END)
    print(f"      Loaded {crypto_px.shape[1]} coins, {crypto_px.shape[0]} days")

    # Pairs selection (in-sample only)
    print("\n[2/5] Selecting cointegrated pairs...")
    in_sample_px = crypto_px.loc[IN_SAMPLE_START:IN_SAMPLE_END]
    final_pairs = find_best_pairs(in_sample_px)
    print(f"      Found {len(final_pairs)} pairs:")
    for pair in final_pairs:
        print(f"        {pair[0]} / {pair[1]}")

    # Generating signals (full sampla)
    print("\n[3/5] Generating signals...")
    signal_df = gen_signals(
        crypto_px,
        final_pairs,
        delta=KALMAN_DELTA,
        R=KALMAN_R,
        zscore_window=ZSCORE_WINDOW
    )
    print(f"      Signal DataFrame shape: {signal_df.shape}")

    # Building the portfolio
    print("\n[4/5] Building portfolio...")
    pos = gen_portfolio(signal_df, final_pairs, crypto_px, threshold=EXIT_THRESHOLD)

    # Trimming to out-of-sample period only
    pos_oos        = pos[pos.index >= OUT_SAMPLE_START]
    crypto_px_oos  = crypto_px[crypto_px.index >= OUT_SAMPLE_START]
    signal_df_oos  = signal_df[signal_df.index >= OUT_SAMPLE_START]

    print(f"      Portfolio shape: {pos_oos.shape}")

    # Performance evaluation
    print("\n[5/5] Evaluating performance...")

    gross_returns, net_returns, turnover = compute_net_returns(
        pos_oos, crypto_px_oos, tcost_bps=TCOST_BPS
    )

    # BTC buy-and-hold benchmark
    btc_returns = crypto_px_oos['bitcoin'].pct_change().dropna()

    print("\n── Strategy Performance ─────────────────────────────────────────")
    compute_stats(net_returns, gross_returns, turnover, benchmark_returns=btc_returns)

    print("\n── Benchmark (Buy & Hold BTC) ───────────────────────────────────")
    compute_stats(btc_returns, btc_returns, pd.Series(0, index=btc_returns.index))


    plot_cumulative_returns(net_returns, benchmark_returns=btc_returns)
    plot_drawdown(net_returns)
    plot_rolling_sharpe(net_returns)


if __name__ == "__main__":
    main()
