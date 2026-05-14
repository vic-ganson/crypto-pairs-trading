import time
from datetime import datetime

import os
import pandas as pd
from pycoingecko import CoinGeckoAPI

cg = CoinGeckoAPI()
CACHE_PATH = "data/crypto_prices.csv"

# Loads prices from cache if possible, otherwise fetches from API.
def load_prices(coin_ids: list, start: str, end: str, force_refresh=False) -> pd.DataFrame:
    
    if os.path.exists(CACHE_PATH) and not force_refresh: # force_refresh = True bypasses the cache
        print("Loading prices from cache...")
        df = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
        return df

    print("Fetching prices from CoinGecko...")
    df = fetch_prices(coin_ids, start, end)

    os.makedirs("data", exist_ok=True)
    df.to_csv(CACHE_PATH)
    print(f"Saved to {CACHE_PATH}")
    return df

# Gets prices from API
def fetch_prices(coin_ids: list, start: str, end: str) -> pd.DataFrame:
    start_ts = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end, "%Y-%m-%d").timestamp())
    
    all_prices = {}

    for coin in coin_ids:
        try:
            data = cg.get_coin_market_chart_range_by_id(
                id=coin,
                vs_currency="usd",
                from_timestamp=start_ts,
                to_timestamp=end_ts
            )
            prices = pd.DataFrame(data["prices"], columns=["timestamp", coin])
            prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
            prices = prices.set_index("timestamp")[coin]
            all_prices[coin] = prices

            time.sleep(1.5)  # respect rate limits

        except Exception as e:
            print(f"Failed to fetch {coin}: {e}")
            continue

    df = pd.DataFrame(all_prices)
    df.index = df.index.normalize()  # strip time, keep date only
    return df
