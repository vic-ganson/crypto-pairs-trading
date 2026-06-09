import os
import pandas as pd
import yfinance as yf

CACHE_PATH = "data/crypto_prices.csv"

COINGECKO_TO_YAHOO = {
    'bitcoin': 'BTC-USD',
    'ethereum': 'ETH-USD',
    'binancecoin': 'BNB-USD',
    'ripple': 'XRP-USD',
    'cardano': 'ADA-USD',
    'solana': 'SOL-USD',
    'polkadot': 'DOT-USD',
    'dogecoin': 'DOGE-USD',
    'avalanche-2': 'AVAX-USD',
    'chainlink': 'LINK-USD',
    'litecoin': 'LTC-USD',
    'uniswap': 'UNI-USD',
    'cosmos': 'ATOM-USD',
    'algorand': 'ALGO-USD',
    'stellar': 'XLM-USD',
    'filecoin': 'FIL-USD',
    'tron': 'TRX-USD',
    'monero': 'XMR-USD',
    'ethereum-classic': 'ETC-USD',
    'vechain': 'VET-USD'
}

def load_prices(coin_ids: list, start: str, end: str, force_refresh=False) -> pd.DataFrame:
    if os.path.exists(CACHE_PATH) and not force_refresh:
        print("Loading prices from cache...")
        df = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
        return df

    print("Fetching prices from Yahoo Finance...")
    df = fetch_prices(coin_ids, start, end)

    os.makedirs("data", exist_ok=True)
    df.to_csv(CACHE_PATH)
    print(f"Saved to {CACHE_PATH}")
    return df

def fetch_prices(coin_ids: list, start: str, end: str) -> pd.DataFrame:
    # Convert CoinGecko IDs to Yahoo tickers
    tickers = [COINGECKO_TO_YAHOO[c] for c in coin_ids if c in COINGECKO_TO_YAHOO]
    missing = [c for c in coin_ids if c not in COINGECKO_TO_YAHOO]
    if missing:
        print(f"Warning: no Yahoo ticker mapping for: {missing}")

    # Fetch all tickers in one call — much faster than one by one
    raw_prices = yf.download(tickers, start=start, end=end, auto_adjust=True)['Close']

    # Rename columns back to CoinGecko IDs for consistency with rest of codebase
    yahoo_to_coingecko = {v: k for k, v in COINGECKO_TO_YAHOO.items()}
    raw_prices = raw_prices.rename(columns=yahoo_to_coingecko)

    return raw_prices
