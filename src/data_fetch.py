from datetime import datetime, timezone
from typing import List

import pandas as pd
from pycoingecko import CoinGeckoAPI

cg = CoinGeckoAPI()


def _date_to_unix_seconds(date_str: str) -> int:
    parsed = datetime.fromisoformat(date_str)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def get_price_data(coin_ids: List[str], start: str, end: str) -> pd.DataFrame:
    # CoinGecko free tier: use market_chart/range endpoint
    # Returns a DataFrame of daily closing prices
    if isinstance(coin_ids, str):
        coin_ids = [coin_ids]

    start_ts = _date_to_unix_seconds(start)
    end_ts = _date_to_unix_seconds(end)

    prices_by_coin = {}
    for coin_id in coin_ids:
        response = cg.get_coin_market_chart_range_by_id(
            id=coin_id,
            vs_currency="usd",
            from_timestamp=start_ts,
            to_timestamp=end_ts,
        )
        prices = response.get("prices", [])
        if not prices:
            raise ValueError(f"No price data returned for {coin_id}")

        df = pd.DataFrame(prices, columns=["timestamp", coin_id])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.date
        daily = df.groupby("date")[coin_id].last()
        prices_by_coin[coin_id] = daily

    price_df = pd.DataFrame(prices_by_coin)
    price_df.index = pd.to_datetime(price_df.index)
    price_df = price_df.sort_index()
    return price_df
