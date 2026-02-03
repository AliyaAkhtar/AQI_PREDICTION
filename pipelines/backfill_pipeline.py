from datetime import datetime, timedelta, timezone
import pandas as pd
from data_sources.pollution_api import fetch_pollution_history
from data_sources.weather_api import fetch_weather_history
from features.preprocessing import clean_data, cap_outliers
from features.feature_engineering import (
    add_time_features,
    add_cyclical_time_features,
    add_lag_features,
    add_rolling_features,
    add_weather_interactions,
    add_future_targets,
    add_real_aqi   
)

from feature_store.mongodb_store import upsert_features
from config.config import CITY, BACKFILL_DAYS

def run_backfill():
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=BACKFILL_DAYS)

    print(f"Backfilling data from {start_dt.date()} to {end_dt.date()}")

    pollution_df = fetch_pollution_history(start_dt, end_dt)
    weather_df = fetch_weather_history(start_dt.date(), end_dt.date())

    df = pd.merge(pollution_df, weather_df, on="timestamp", how="inner")
    df["city"] = CITY

    # Preprocessing
    df = clean_data(df)
    df = add_real_aqi(df)  
    df = cap_outliers(df)

    # Feature Engineering
    df = add_time_features(df)
    df = add_cyclical_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_weather_interactions(df)
    df = add_future_targets(df)

    # Insert all rows
    upsert_features(df)
    print("Backfill completed successfully!")

if __name__ == "__main__":
    run_backfill()
