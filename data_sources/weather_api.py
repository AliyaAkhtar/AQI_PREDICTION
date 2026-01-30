import requests
import pandas as pd
from config.config import LAT, LON


def fetch_weather_history(start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relativehumidity_2m,pressure_msl,windspeed_10m",
        "timezone": "UTC"
    }

    res = requests.get(url, params=params)
    res.raise_for_status()

    data = res.json()["hourly"]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    df.drop(columns=["time"], inplace=True)

    return df