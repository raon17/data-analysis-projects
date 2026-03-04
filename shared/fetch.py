"""
Generic data fetching functions used across all projects
Each project-specific API will get its own function here and return raw data 
"""

import os
import requests
from dotenv import load_dotenv

def get_steam_data(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()