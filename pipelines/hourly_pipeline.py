from datetime import datetime, timedelta, timezone
import pandas as pd

from data_sources.pollution_api import fetch_pollution_history
from data_sources.weather_api import fetch_weather_history
from features.preprocessing import clean_data
from features.feature_engineering import add_time_features
from feature_store.mongodb_store import upsert_features
from config.config import CITY


def run_hourly():
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=1)

    pollution_df = fetch_pollution_history(start_dt, end_dt)
    weather_df = fetch_weather_history(start_dt.date(), end_dt.date())

    df = pd.merge(pollution_df, weather_df, on="timestamp", how="inner")
    df["city"] = CITY

    df = clean_data(df)
    df = add_time_features(df)

    upsert_features(df)
    print("Hourly update done")


if __name__ == "__main__":
    run_hourly()