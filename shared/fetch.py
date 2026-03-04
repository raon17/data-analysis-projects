#Generic data fetching functions used across all projects
#Each project-specific API will get its own function here and return raw data 

import os 
import requests 
from dotenv import load_dotenv 

def get_json(url: str, params: dict = None) -> dict:
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise an error for bad status codes ()
    return response.json()

def get_json_list(url: str, params: dict = None) -> list:
    data = get_json(url, params)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected a list but data returned is {type(data)}")

#Used for fetching stock data from Alpha Vantage API, used in Project Crypto-Analysis
def get_alpha_vantage_key() -> str:
    key = os.getenv("ALPHA_VANTAGE_KEY")
    if not key:
        raise EnvironmentError(
            "Alpha Vantage API key not found.\n"
        )
    return key

#Fetch daily OHLCV data Alpha Vantage API, used in Project Crypto-Analysis
def get_crypto_daily(symbol: str, market: str = "USD") -> dict:
    url = "https://www.alphavantage.co/query"
    params = {
        "function":"DIGITAL_CURRENCY_DAILY",
        "symbol":symbol,
        "market":market,
        "apikey":get_alpha_vantage_key(),
        "outputsize":"full",
    }

    data = get_json(url, params)

    # Error handling for API responses
    if "Error Message" in data:
        raise ValueError(f"Error {symbol}: {data['Error Message']}")

    if "Note" in data:
        raise RuntimeError(
            f"Limit hit for {symbol}.\n, free tier allows 25 requests/day. Wait and try again.")

    if "Information" in data:
        raise RuntimeError(
            f"Information for {symbol}: {data['Information']}"
        )
    return data

def get_multiple_crypto(symbols: list, market: str = "USD") -> dict: 
    results = {}
    for symbol in symbols:
        print(f"  Fetching {symbol}...", end=" ")
        try:
            results[symbol] = get_crypto_daily(symbol, market)
            print("Done.")
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
    return results