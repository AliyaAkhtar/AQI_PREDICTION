import os
import mlflow
import mlflow.sklearn
import pandas as pd
from datetime import timedelta
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# ================== CONFIG ==================
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = "aqi_prediction"
FEATURES_COLLECTION = "features_karachi_hourly"
PREDICTIONS_COLLECTION = "aqi_forecasts_daily"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
features_col = db[FEATURES_COLLECTION]
preds_col = db[PREDICTIONS_COLLECTION]


# ================== LOAD PRODUCTION MODEL ==================
def load_production_model():
    model_uri = "models:/AQI_Forecast_Model/Production"
    print(f"Loading PRODUCTION model: {model_uri}")

    model = mlflow.sklearn.load_model(model_uri)

    # Load model signature (feature names)
    model_meta = mlflow.models.Model.load(model_uri)
    signature = model_meta.signature

    if signature is None:
        raise ValueError("Model signature missing! Re-log model with signature.")

    feature_names = [col.name for col in signature.inputs]

    return model, feature_names


# ================== LOAD LATEST FEATURES ==================
def get_latest_features():
    df = pd.DataFrame(list(features_col.find().sort("timestamp", -1).limit(1)))

    if df.empty:
        raise ValueError("No feature data found in MongoDB.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# ================== FEATURE CLEANING ==================
def prepare_features_for_model(df, feature_names):
    # Keep only required columns
    X = df.reindex(columns=feature_names)

    # Handle missing values (important for lag/rolling features)
    X = X.fillna(method="ffill").fillna(method="bfill").fillna(0)

    return X


# ================== PREDICTION ==================
def predict_next_3_days(model, feature_names, latest_df):
    X = prepare_features_for_model(latest_df, feature_names)

    preds = model.predict(X)[0]  # [24h, 48h, 72h]

    today = pd.Timestamp.utcnow().normalize()

    results = pd.DataFrame({
        "date": [
            today + timedelta(days=1),
            today + timedelta(days=2),
            today + timedelta(days=3),
        ],
        "avg_aqi": preds
    })

    return results


# ================== CHECK CACHE ==================
def check_existing_predictions():
    today = pd.Timestamp.utcnow().normalize()
    return pd.DataFrame(list(preds_col.find({"date": {"$gte": today}})))


# ================== RUN INFERENCE ==================
def run_inference():
    existing = check_existing_predictions()
    if len(existing) >= 3:
        print("Using cached predictions")
        return existing

    model, feature_names = load_production_model()
    latest_df = get_latest_features()

    daily_preds = predict_next_3_days(model, feature_names, latest_df)

    preds_col.delete_many({})
    preds_col.insert_many(daily_preds.to_dict("records"))

    print("Saved new AQI forecast")
    return daily_preds


if __name__ == "__main__":
    print(run_inference())
