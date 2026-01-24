import requests
import pandas as pd
from feature_store.hopsworks_utils import get_feature_group
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv
import time

load_dotenv()

CITY = os.getenv("CITY")
LAT = float(os.getenv("LAT"))
LON = float(os.getenv("LON"))
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

MAX_RETRIES = 3
RETRY_DELAY = 60  

def get_previous_hour_timestamp():
    now = datetime.now(timezone.utc)
    prev_hour = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    return prev_hour

def fetch_current_pollution(prev_hour_ts):
    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    start_unix = int(prev_hour_ts.timestamp())
    end_unix = start_unix

    params = {
        "lat": LAT,
        "lon": LON,
        "start": start_unix,
        "end": end_unix,
        "appid": OPENWEATHER_API_KEY
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json().get("list", [])

    if not data:
        return pd.DataFrame()

    item = data[0]
    ts = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
    comp = item["components"]

    return pd.DataFrame([{
        "timestamp": ts,
        "pm2_5": comp.get("pm2_5"),
        "pm10": comp.get("pm10"),
        "no2": comp.get("no2"),
        "so2": comp.get("so2"),
        "o3": comp.get("o3"),
        "co": comp.get("co"),
        "us_aqi": item.get("main", {}).get("aqi")
    }])

def fetch_current_weather(prev_hour_ts):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": [
            "temperature_2m", "relativehumidity_2m",
            "pressure_msl", "windspeed_10m"
        ],
        "timezone": "UTC"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    df = pd.DataFrame(response.json().get("hourly", {}))
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    df.drop(columns=["time"], inplace=True)

    return df[df["timestamp"] == prev_hour_ts]

def update_feature_store():
    fg = get_feature_group()
    prev_hour_ts = get_previous_hour_timestamp()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            pollution_df = fetch_current_pollution(prev_hour_ts)
            weather_df = fetch_current_weather(prev_hour_ts)

            if pollution_df.empty:
                raise ValueError("No pollution/AQI data")
            if weather_df.empty:
                raise ValueError("No weather data")

            break
        except Exception as e:
            print(f"Attempt {attempt}: {e}")
            if attempt < MAX_RETRIES:
                print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"Failed to fetch data after {MAX_RETRIES} attempts. Skipping this hour.")
                return

    df = pd.merge(pollution_df, weather_df, on="timestamp", how="inner")
    df["city"] = CITY

    df["hour"] = df["timestamp"].dt.hour.astype("int32")
    df["day_of_week"] = df["timestamp"].dt.weekday.astype("int32")
    df["is_weekend"] = (df["timestamp"].dt.weekday >= 5).astype("int32")

    if "relativehumidity_2m" in df.columns:
        df["relativehumidity_2m"] = pd.to_numeric(
            df["relativehumidity_2m"], errors="coerce"
        ).fillna(0).astype("Int64")

    float_cols = [
        "temperature_2m", "pressure_msl", "windspeed_10m",
        "pm2_5", "pm10", "no2", "so2", "o3", "co", "us_aqi"
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

    try:
        existing_df = fg.read().to_pandas()
        key_cols = ["timestamp"] + float_cols + ["relativehumidity_2m"]

        new_rows = df.merge(
            existing_df[key_cols],
            on=key_cols,
            how='left',
            indicator=True
        ).query('_merge == "left_only"').drop(columns=["_merge"])
    except Exception:
        print("Feature group empty or unreadable, inserting data.")
        new_rows = df

    if new_rows.empty:
        print(f"No new data to insert for {prev_hour_ts}.")
        return

    new_rows = new_rows[[
        "city", "timestamp",
        "temperature_2m", "relativehumidity_2m", "pressure_msl", "windspeed_10m",
        "pm2_5", "pm10", "no2", "so2", "o3", "co",
        "us_aqi",
        "hour", "day_of_week", "is_weekend"
    ]]

    fg.insert(new_rows, write_options={"wait_for_job": True})
    print(f"Inserted {len(new_rows)} row(s) into Hopsworks for {prev_hour_ts}.")

if __name__ == "__main__":
    update_feature_store()
