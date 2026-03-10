
import pandas as pd
import numpy as np

def parse_coin(raw: dict, symbol: str) -> pd.DataFrame:
    TIME_SERIES_KEY = "Time Series (Digital Currency Daily)"
    if TIME_SERIES_KEY not in raw:
        raise KeyError(
            f"Expected key '{TIME_SERIES_KEY}' not found for {symbol}.\n"
            f"Keys in response: {list(raw.keys())}"
        )
    time_series = raw[TIME_SERIES_KEY]
    rows = []
    for date_str, values in time_series.items():
        rows.append({
            "date":   date_str,
            "symbol": symbol,
            "open":   float(values.get("1a. open (USD)",  values.get("1. open",  0))),
            "high":   float(values.get("2a. high (USD)",  values.get("2. high",  0))),
            "low":    float(values.get("3a. low (USD)",   values.get("3. low",   0))),
            "close":  float(values.get("4a. close (USD)", values.get("4. close", 0))),
            "volume": float(values.get("5. volume", 0)),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    # API returns newest-first — we want oldest-first for time series
    df = df.sort_values("date").reset_index(drop=True)

    return df


def parse_coins(raw_data: dict) -> pd.DataFrame:
    parsed = []

    for symbol, raw in raw_data.items():
        print(f"  Parsing {symbol}...", end=" ", flush=True)
        try:
            df = parse_coin(raw, symbol)
            parsed.append(df)
            print(f"✅  ({len(df):,} days)")
        except Exception as e:
            print(f"❌  {e}")

    if not parsed:
        raise ValueError("No coins were successfully parsed.")

    # Stack all coin DataFrames vertically into one
    combined = pd.concat(parsed, ignore_index=True)
    combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)

    return combined
