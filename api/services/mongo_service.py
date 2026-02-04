from pymongo import MongoClient
import pandas as pd
from datetime import timedelta
from config.config import MONGO_URI, MONGO_DB

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]

preds_col = db["aqi_forecasts_daily"]
features_col = db["features_karachi_hourly"]

def get_forecasts():
    today = pd.Timestamp.utcnow().normalize()
    data = list(preds_col.find({"date": {"$gte": today}}, {"_id": 0}))
    return data

def get_history(days: int):
    end = pd.Timestamp.utcnow()
    start = end - timedelta(days=days)

    data = list(features_col.find(
        {"timestamp": {"$gte": start}},
        {"_id": 0, "timestamp": 1, "real_aqi": 1}
    ).sort("timestamp", 1))

    return data
