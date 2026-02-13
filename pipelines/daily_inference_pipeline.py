# import os
# import mlflow
# import mlflow.sklearn
# import pandas as pd
# from datetime import timedelta
# from pymongo import MongoClient
# from dotenv import load_dotenv
# from mlflow.tracking import MlflowClient

# load_dotenv()

# #  CONFIG 
# MONGO_URI = os.getenv("MONGO_URI")
# MONGO_DB = "aqi_prediction"
# FEATURES_COLLECTION = "features_karachi_hourly"
# PREDICTIONS_COLLECTION = "aqi_forecasts_daily"

# mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
# os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
# os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# client = MongoClient(MONGO_URI)
# db = client[MONGO_DB]
# features_col = db[FEATURES_COLLECTION]
# preds_col = db[PREDICTIONS_COLLECTION]

# #  LOAD PRODUCTION MODEL 
# def load_production_model():
#     model_name = "AQI_Forecast_Model"
#     client = MlflowClient()

#     # Get Production version
#     for mv in client.search_model_versions(f"name='{model_name}'"):
#         if mv.current_stage == "Production":
#             model_uri = f"models:/{model_name}/{mv.version}"
#             break
#     else:
#         raise ValueError("No Production model found!")

#     print(f"Loading PRODUCTION model version {mv.version}")

#     model = mlflow.sklearn.load_model(model_uri)

#     # Load signature from model artifact
#     local_path = mlflow.artifacts.download_artifacts(model_uri)
#     model_meta = mlflow.models.Model.load(os.path.join(local_path, "MLmodel"))
#     signature = model_meta.signature

#     if signature is None:
#         raise ValueError("Model signature missing! Re-log model with signature.")

#     feature_names = [col.name for col in signature.inputs]
#     return model, feature_names

# #  LOAD LATEST FEATURES 
# def get_latest_features():
#     df = pd.DataFrame(list(features_col.find().sort("timestamp", -1).limit(1)))

#     if df.empty:
#         raise ValueError("No feature data found in MongoDB.")

#     df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
#     return df

# #  FEATURE CLEANING 
# def prepare_features_for_model(df, feature_names):
#     missing = set(feature_names) - set(df.columns)
#     if missing:
#         print(f"Warning: Missing features filled with 0 → {missing}")

#     X = df.reindex(columns=feature_names)

#     # Handle missing values 
#     X = X.fillna(method="ffill").fillna(method="bfill").fillna(0)

#     return X

# #  PREDICTION 
# def predict_next_3_days(model, feature_names, latest_df):
#     X = prepare_features_for_model(latest_df, feature_names)

#     preds = model.predict(X)[0]  

#     today = pd.Timestamp.utcnow().normalize()

#     results = pd.DataFrame({
#         "date": [
#             today + timedelta(days=1),
#             today + timedelta(days=2),
#             today + timedelta(days=3),
#         ],
#         "avg_aqi": preds
#     })

#     return results

# #  CHECK CACHE 
# def check_existing_predictions():
#     today = pd.Timestamp.utcnow().normalize().to_pydatetime()
#     existing = list(preds_col.find({"date": {"$gte": today}}).sort("date", 1))
#     return pd.DataFrame(existing)

# def run_inference():
#     existing = check_existing_predictions()
#     if len(existing) >= 3:
#         print("Using cached predictions")
#         return existing

#     model, feature_names = load_production_model()
#     latest_df = get_latest_features()

#     daily_preds = predict_next_3_days(model, feature_names, latest_df)

#     daily_preds["date"] = pd.to_datetime(daily_preds["date"]).dt.to_pydatetime()

#     today = pd.Timestamp.utcnow().normalize().to_pydatetime()
#     preds_col.delete_many({"date": {"$gte": today}})

#     preds_col.insert_many(daily_preds.to_dict("records"))

#     print("Saved new AQI forecast")
#     return daily_preds

# if __name__ == "__main__":
#     print(run_inference())

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

load_dotenv()

# ================== CONFIG ==================
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = "aqi_prediction"
FEATURES_COLLECTION = "features_karachi_hourly"
PREDICTIONS_COLLECTION = "aqi_forecasts_daily"

CITY = "Karachi"
LAT = "24.8607"
LON = "67.0011"
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
features_col = db[FEATURES_COLLECTION]
preds_col = db[PREDICTIONS_COLLECTION]

# ================== LOAD PRODUCTION MODEL ==================
def load_production_model():
    model_name = "AQI_Forecast_Model"
    mlflow_client = MlflowClient()

    # Get Production version
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        if mv.current_stage == "Production":
            model_uri = f"models:/{model_name}/{mv.version}"
            break
    else:
        raise ValueError("No Production model found!")

    print(f"Loading PRODUCTION model version {mv.version}")
    model = mlflow.sklearn.load_model(model_uri)

    # Load model signature
    local_path = mlflow.artifacts.download_artifacts(model_uri)
    model_meta = mlflow.models.Model.load(os.path.join(local_path, "MLmodel"))
    signature = model_meta.signature
    if signature is None:
        raise ValueError("Model signature missing!")

    feature_names = [col.name for col in signature.inputs]
    return model, feature_names

# ================== FETCH FUTURE WEATHER ==================
def fetch_weather_forecast():
    """
    Fetch weather forecast for next 72 hours
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m,relativehumidity_2m,pressure_msl,windspeed_10m",
        "forecast_days": 3,
        "timezone": "UTC"
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    df.drop(columns=["time"], inplace=True)
    return df

# ================== GENERATE FUTURE FEATURES ==================
def generate_future_features(latest_df, hours=72):
    """
    Generate future hourly features for next N hours
    """
    last_timestamp = latest_df["timestamp"].max()
    future_times = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=hours,
        freq="H",
        tz="UTC"
    )

    # Fetch weather forecast
    weather_df = fetch_weather_forecast()
    weather_future = weather_df[weather_df["timestamp"].isin(future_times)]

    # Create placeholder pollution columns (if no forecast, carry last known)
    pollutant_cols = ["pm2_5", "pm10", "no2", "so2", "o3", "co", "real_aqi"]
    last_pollution = latest_df[pollutant_cols].iloc[-1]
    pollution_future = pd.DataFrame([last_pollution.values] * len(future_times), columns=pollutant_cols)
    pollution_future["timestamp"] = future_times

    # Combine weather + pollutants
    future_df = pd.merge_asof(
        pollution_future.sort_values("timestamp"),
        weather_future.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("30min")
    )

    # Add time features
    future_df["hour"] = future_df["timestamp"].dt.hour
    future_df["day_of_week"] = future_df["timestamp"].dt.weekday
    future_df["day_of_month"] = future_df["timestamp"].dt.day
    future_df["month"] = future_df["timestamp"].dt.month
    future_df["quarter"] = (future_df["timestamp"].dt.month - 1) // 3 + 1
    future_df["is_weekend"] = (future_df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding
    future_df["hour_sin"] = np.sin(2 * np.pi * future_df["hour"] / 24)
    future_df["hour_cos"] = np.cos(2 * np.pi * future_df["hour"] / 24)
    future_df["day_of_week_sin"] = np.sin(2 * np.pi * future_df["day_of_week"] / 7)
    future_df["day_of_week_cos"] = np.cos(2 * np.pi * future_df["day_of_week"] / 7)
    future_df["month_sin"] = np.sin(2 * np.pi * future_df["month"] / 12)
    future_df["month_cos"] = np.cos(2 * np.pi * future_df["month"] / 12)

    return future_df

# ================== PREDICT NEXT 3 DAYS ==================
def predict_next_3_days(model, feature_names, latest_df):
    # Generate future hourly features
    future_df = generate_future_features(latest_df, hours=72)

    # Prepare features
    X = future_df.reindex(columns=feature_names).ffill().bfill().fillna(0)

    # Predict hourly AQI
    hourly_preds = model.predict(X)

    if len(hourly_preds.shape) > 1:
        hourly_preds = hourly_preds.mean(axis=1)
    future_df["predicted_aqi"] = np.clip(hourly_preds, 0, None)

    # Aggregate hourly → daily
    future_df["date"] = future_df["timestamp"].dt.normalize()
    daily_avg = (
        future_df.groupby("date")["predicted_aqi"]
        .mean()
        .reset_index()
        .rename(columns={"predicted_aqi": "avg_aqi"})
    )

    # # Ensure tz-aware
    # if daily_avg["date"].dt.tz is None:
    #     daily_avg["date"] = daily_avg["date"].dt.tz_localize("UTC")

    # Take only next 3 days excluding today
    # today = pd.Timestamp.utcnow().normalize().tz_localize("UTC")
    # daily_avg = daily_avg[daily_avg["date"] > today].head(3)

    # Take only next 3 days excluding today
    today = pd.Timestamp.utcnow().normalize()
    if today.tzinfo is None:
        today = today.tz_localize("UTC")

    daily_avg = daily_avg[daily_avg["date"] > today].head(3)

    return daily_avg

# ================== LOAD LATEST FEATURES ==================
def get_latest_features():
    df = pd.DataFrame(list(features_col.find().sort("timestamp", 1)))
    if df.empty:
        raise ValueError("No feature data found in MongoDB.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

# ================== CHECK EXISTING PREDICTIONS ==================
# def check_existing_predictions():
#     today = pd.Timestamp.utcnow().normalize()
#     if today.tzinfo is None:  # only localize if naive
#         today = today.tz_localize("UTC")
#     existing = list(preds_col.find({"date": {"$gte": today}}).sort("date", 1))
#     return pd.DataFrame(existing)
def check_existing_predictions():
    today = pd.Timestamp.utcnow().normalize()
    if today.tzinfo is None:
        today = today.tz_localize("UTC")

    existing = list(
        preds_col.find({"date": {"$gt": today}}).sort("date", 1)
    )
    return pd.DataFrame(existing)

# ================== RUN INFERENCE ==================
# def run_inference():
#     existing = check_existing_predictions()
#     if len(existing) >= 3:
#         print("Using cached predictions")
#         return existing

#     model, feature_names = load_production_model()
#     latest_df = get_latest_features()
#     daily_preds = predict_next_3_days(model, feature_names, latest_df)
#     if daily_preds.empty:
#         print("No new predictions available.")
#         return daily_preds

#     # Convert date column → python datetime for MongoDB
#     daily_preds["date"] = pd.to_datetime(daily_preds["date"]).dt.tz_convert(None)

#     # Delete existing predictions for next 3 days
#     today = pd.Timestamp.utcnow().normalize()
#     preds_col.delete_many({"date": {"$gte": today}})

#     # Insert new predictions safely
#     records = daily_preds.to_dict("records")
#     if records:
#         preds_col.insert_many(records)
#         print("Saved new AQI forecast")
#     else:
#         print("No records to save")

#     return daily_preds

def run_inference():
    today = pd.Timestamp.utcnow().normalize()

    # Get existing future predictions
    existing = list(
        preds_col.find({"date": {"$gt": today}}).sort("date", 1)
    )
    existing_df = pd.DataFrame(existing)

    existing_dates = set()
    if not existing_df.empty:
        existing_df["date"] = pd.to_datetime(existing_df["date"]).dt.normalize()
        existing_dates = set(existing_df["date"])

    # Required future dates (next 3 days strictly future)
    target_dates = [
        today + timedelta(days=1),
        today + timedelta(days=2),
        today + timedelta(days=3),
    ]

    missing_dates = [d for d in target_dates if d not in existing_dates]

    if not missing_dates:
        print("Using cached predictions")
        return existing_df

    print(f"Need to predict: {missing_dates}")

    # Load model once
    model, feature_names = load_production_model()
    latest_df = get_latest_features()

    # We only predict up to the furthest missing day
    max_missing_day = max(missing_dates)
    days_ahead = (max_missing_day - today).days

    # Generate hourly predictions only for required horizon
    future_df = generate_future_features(latest_df, hours=24 * days_ahead)

    X = future_df.reindex(columns=feature_names).ffill().bfill().fillna(0)
    # hourly_preds = model.predict(X)
    # future_df["predicted_aqi"] = np.clip(hourly_preds, 0, None)

    hourly_preds = model.predict(X)

    # Fix for multi-output model
    if len(hourly_preds.shape) > 1:
        hourly_preds = hourly_preds.mean(axis=1)

    hourly_preds = np.clip(hourly_preds, 0, None)
    future_df["predicted_aqi"] = hourly_preds

    # Aggregate daily
    future_df["date"] = future_df["timestamp"].dt.normalize()
    daily_avg = (
        future_df.groupby("date")["predicted_aqi"]
        .mean()
        .reset_index()
        .rename(columns={"predicted_aqi": "avg_aqi"})
    )

    # Keep only missing ones
    daily_avg = daily_avg[daily_avg["date"].isin(missing_dates)]

    if not daily_avg.empty:
        daily_avg["date"] = pd.to_datetime(daily_avg["date"]).dt.to_pydatetime()
        preds_col.insert_many(daily_avg.to_dict("records"))
        print("Inserted only missing forecast days")

    # Return updated 3-day window
    final = list(
        preds_col.find({"date": {"$gt": today}}).sort("date", 1)
    )
    return pd.DataFrame(final)

# ================== MAIN ==================
if __name__ == "__main__":
    print(run_inference())
