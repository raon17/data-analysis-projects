import requests

def get_steam_data(url):
    response = requests.get(url)
    response.raise_for_status()  # throws error if request failed
    return response.json()