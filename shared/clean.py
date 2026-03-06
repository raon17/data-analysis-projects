import pandas as pd
import numpy as np

def remove_nulls(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    return df

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = normalise_columns(df)
    df = drop_duplicates(df)
    df = remove_nulls(df)
    return df

def parse_crypto(raw: dict, symbol: str) -> pd.DataFrame:
    time_series_key = "Time Series (Digital Currency Daily)"

    if time_series_key not in raw:
        raise KeyError(
            f"Expected key '{time_series_key}' not found in response for {symbol}.\n"
            f"Keys present: {list(raw.keys())}"
        )

    time_series = raw[time_series_key]