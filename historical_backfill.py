import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from feature_store.hopsworks_utils import get_feature_group
from dotenv import load_dotenv
import os
import time

load_dotenv()

CITY = os.getenv("CITY")
LAT = float(os.getenv("LAT"))
LON = float(os.getenv("LON"))
BACKFILL_DAYS = int(os.getenv("BACKFILL_DAYS"))
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


def fetch_pollution_history(start_unix, end_unix):
    """
    Fetch historical pollutants + AQI from OpenWeatherMap Air Pollution API
    """
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
        row = {
            "timestamp": ts,
            "pm2_5": comp.get("pm2_5", None),
            "pm10": comp.get("pm10", None),
            "no2": comp.get("no2", None),
            "so2": comp.get("so2", None),
            "o3": comp.get("o3", None),
            "co": comp.get("co", None),
            "us_aqi": item.get("main", {}).get("aqi", None)  
        }
        rows.append(row)
    return pd.DataFrame(rows)


def fetch_weather_history(start_date, end_date):
    """
    Fetch historical weather (temperature, humidity, wind, pressure) from Open-Meteo
    Supports multi-day range.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
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


def backfill():
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=BACKFILL_DAYS)
    print(f"Fetching historical data from {start_date} → {end_date}")

    # --- Fetch AQI + pollutants from OpenWeather ---
    start_unix = int(datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc).timestamp())
    end_unix = int(datetime.combine(end_date, datetime.min.time(), tzinfo=timezone.utc).timestamp())
    pollution_df = fetch_pollution_history(start_unix, end_unix)

    # --- Fetch weather from Open-Meteo ---
    weather_df = fetch_weather_history(start_date.isoformat(), end_date.isoformat())

    # --- Merge weather + pollution data on timestamp ---
    df = pd.merge(pollution_df, weather_df, on="timestamp", how="inner")

    # Add city and time features
    df["city"] = CITY
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.weekday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Reorder columns for Hopsworks
    df = df[[
        "city", "timestamp",
        "temperature_2m", "relativehumidity_2m", "pressure_msl", "windspeed_10m",
        "pm2_5", "pm10", "no2", "so2", "o3", "co",
        "us_aqi",
        "hour", "day_of_week", "is_weekend"
    ]]

    # Insert into Hopsworks
    fg = get_feature_group()
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"Inserted {len(df)} historical rows into Hopsworks")


if __name__ == "__main__":
    backfill()
