import os
import mlflow
import mlflow.sklearn
import pandas as pd
from datetime import timedelta
from pymongo import MongoClient
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()

#  CONFIG 
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

#  LOAD PRODUCTION MODEL 
def load_production_model():
    model_name = "AQI_Forecast_Model"
    client = MlflowClient()

    # Get Production version
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if mv.current_stage == "Production":
            model_uri = f"models:/{model_name}/{mv.version}"
            break
    else:
        raise ValueError("No Production model found!")

    print(f"Loading PRODUCTION model version {mv.version}")

    model = mlflow.sklearn.load_model(model_uri)

    # Load signature from model artifact
    local_path = mlflow.artifacts.download_artifacts(model_uri)
    model_meta = mlflow.models.Model.load(os.path.join(local_path, "MLmodel"))
    signature = model_meta.signature

    if signature is None:
        raise ValueError("Model signature missing! Re-log model with signature.")

    feature_names = [col.name for col in signature.inputs]
    return model, feature_names

#  LOAD LATEST FEATURES 
def get_latest_features():
    df = pd.DataFrame(list(features_col.find().sort("timestamp", -1).limit(1)))

    if df.empty:
        raise ValueError("No feature data found in MongoDB.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df

#  FEATURE CLEANING 
def prepare_features_for_model(df, feature_names):
    missing = set(feature_names) - set(df.columns)
    if missing:
        print(f"Warning: Missing features filled with 0 → {missing}")

    X = df.reindex(columns=feature_names)

    # Handle missing values 
    X = X.fillna(method="ffill").fillna(method="bfill").fillna(0)

    return X

#  PREDICTION 
def predict_next_3_days(model, feature_names, latest_df):
    X = prepare_features_for_model(latest_df, feature_names)

    preds = model.predict(X)[0]  

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

#  CHECK CACHE 
def check_existing_predictions():
    today = pd.Timestamp.utcnow().normalize().to_pydatetime()
    existing = list(preds_col.find({"date": {"$gte": today}}).sort("date", 1))
    return pd.DataFrame(existing)

def run_inference():
    existing = check_existing_predictions()
    if len(existing) >= 3:
        print("Using cached predictions")
        return existing

    model, feature_names = load_production_model()
    latest_df = get_latest_features()

    daily_preds = predict_next_3_days(model, feature_names, latest_df)

    daily_preds["date"] = pd.to_datetime(daily_preds["date"]).dt.to_pydatetime()

    today = pd.Timestamp.utcnow().normalize().to_pydatetime()
    preds_col.delete_many({"date": {"$gte": today}})

    preds_col.insert_many(daily_preds.to_dict("records"))

    print("Saved new AQI forecast")
    return daily_preds

if __name__ == "__main__":
    print(run_inference())
