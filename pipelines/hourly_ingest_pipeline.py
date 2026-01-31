import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

from features.preprocessing import clean_data, cap_outliers
from features.feature_engineering import (
    add_time_features,
    add_cyclical_time_features,
    add_lag_features,
    add_rolling_features,
    add_weather_interactions,
    add_future_targets
)
from feature_store.mongodb_store import upsert_features

load_dotenv()

CITY = os.getenv("CITY")
LAT = float(os.getenv("LAT"))
LON = float(os.getenv("LON"))
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

def fetch_pollution_last_hour(start_unix, end_unix):
    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        "lat": LAT,
        "lon": LON,
        "start": start_unix,
        "end": end_unix,
        "appid": OPENWEATHER_API_KEY
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()["list"]

    rows = []
    for item in data:
        ts = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
        comp = item["components"]

        rows.append({
            "timestamp": ts,
            "pm2_5": comp.get("pm2_5"),
            "pm10": comp.get("pm10"),
            "no2": comp.get("no2"),
            "so2": comp.get("so2"),
            "o3": comp.get("o3"),
            "co": comp.get("co"),
            "us_aqi": item.get("main", {}).get("aqi")
        })

    return pd.DataFrame(rows)

def fetch_weather_last_hour(date_str):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": "temperature_2m,relativehumidity_2m,pressure_msl,windspeed_10m",
        "timezone": "UTC"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()["hourly"]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    df.drop(columns=["time"], inplace=True)

    return df

def run_hourly_ingestion():
    print("Running hourly AQI ingestion...")

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    end_time = now
    start_time = now - timedelta(hours=1)

    print(f"Fetching data for window: {start_time} → {end_time}")

    pollution_df = fetch_pollution_last_hour(
        int(start_time.timestamp()),
        int(end_time.timestamp())
    )

    if pollution_df.empty:
        print("No pollution data returned. Skipping...")
        return

    weather_df = fetch_weather_last_hour(start_time.date().isoformat())

    # Merge
    df = pd.merge(pollution_df, weather_df, on="timestamp", how="inner")

    df["city"] = CITY

    df = clean_data(df)
    df = cap_outliers(df)

    df = add_time_features(df)
    df = add_cyclical_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_weather_interactions(df)
    df = add_future_targets(df)

    upsert_features(df)

    print(f"Inserted/Updated {len(df)} hourly records successfully!")

if __name__ == "__main__":
    run_hourly_ingestion()
