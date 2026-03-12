#Generic data fetching functions used across all projects
#Each project-specific API will get its own function here and return raw data 

import os 
import requests 
from dotenv import load_dotenv 

def get_api_key() -> str:
    key = os.getenv("ALPHA_VANTAGE_KEY")
    if not key:
        raise EnvironmentError(
            "\n  API key not found."
        )
    return key

def get_json(url: str, params: dict = None) -> dict:
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def check_for_api_errors(data: dict, symbol: str) -> None:
    if "Error Message" in data:
        raise ValueError(
            f"\n request rejected for {symbol}"
        )
    if "Note" in data:
        raise RuntimeError(
            f"\n Limit hit fetching {symbol}."
        )
    if "Information" in data:
        raise RuntimeError(
            f"\nℹ Returned an info message for {symbol}:"
            f"\n{data['Information']}"
        )

def fetch_coin(symbol: str, market: str = "USD") -> dict:
    url = "https://www.alphavantage.co/query"
    params = {
        "function":   "DIGITAL_CURRENCY_DAILY",
        "symbol":     symbol,
        "market":     market,
        "apikey":     get_api_key(),
        "outputsize": "full",
    }
    data = get_json(url, params)
    check_for_api_errors(data, symbol)
    return data


def fetch_coins(symbols: list, market: str = "USD") -> dict:
    results = {}
    for symbol in symbols:
        print(f"  Fetching {symbol}...", end=" ", flush=True)
        try:
            results[symbol] = fetch_coin(symbol, market)
            print("Good")
        except Exception as e:
            print(f"\n{e}")
    print(f"\n  complete - fetched {len(results)}/{len(symbols)} coins")
    return results