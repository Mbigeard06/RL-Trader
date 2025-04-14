import os
from dotenv import load_dotenv
import requests
import pandas as pd

load_dotenv()  # Charge les variables depuis le fichier .env

api_key = os.getenv("TWELVE_API_KEY")

def fetch_twelve_data(ticker, interval="1day", start_date=None, end_date=None):
    url = f"https://api.twelvedata.com/time_series"
    params = {
        "symbol": ticker,
        "interval": interval,
        "apikey": api_key,
        "outputsize": 5000,  
        "format": "JSON"
    }
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    r = requests.get(url, params=params)
    data = r.json()

    if "values" not in data:
        print("Erreur:", data)
        return pd.DataFrame()

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

df = fetch_twelve_data("WMT")
df.to_csv("./data/blueChips/WMT.csv", index=False)
