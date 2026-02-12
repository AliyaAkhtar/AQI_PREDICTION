from pymongo import MongoClient
import pandas as pd
from datetime import timedelta
from config.config import MONGO_URI, MONGO_DB
from datetime import timezone

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]

preds_col = db["aqi_forecasts_daily"]
features_col = db["features_karachi_hourly"]

def get_forecasts():
    today = pd.Timestamp.utcnow().normalize()
    data = list(preds_col.find({"date": {"$gte": today}}, {"_id": 0}))
    return data

# def get_history(days: int):
#     end = pd.Timestamp.utcnow().replace(tzinfo=timezone.utc)
#     start = end - timedelta(days=days)

#     print("Query start:", start, "Query end:", end)

#     data = list(features_col.find(
#         {"timestamp": {"$gte": start}},
#         {"_id": 0, "timestamp": 1, "real_aqi": 1}
#     ).sort("timestamp", 1))

#     return data

def get_history(days: int):
    end = pd.Timestamp.utcnow().replace(tzinfo=timezone.utc)
    start = end - timedelta(days=days)

    data = list(features_col.find(
        {"timestamp": {"$gte": start}},
        {"_id": 0, "timestamp": 1, "real_aqi": 1}
    ))

    if not data:
        return {"history": []}

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.date

    # 🔥 DAILY AVERAGE
    daily_avg = (
        df.groupby("date")["real_aqi"]
        .mean()
        .reset_index()
    )

    daily_avg["real_aqi"] = daily_avg["real_aqi"].round(2)

    return {"history": daily_avg.to_dict(orient="records")}

def get_today_avg_aqi():
    now = pd.Timestamp.utcnow().replace(tzinfo=timezone.utc)

    start_of_today = now.normalize()   # 00:00 UTC today
    end_of_today = start_of_today + pd.Timedelta(days=1)

    print("Today's AQI range:", start_of_today, "to", end_of_today)

    data = list(features_col.find(
        {
            "timestamp": {"$gte": start_of_today, "$lt": end_of_today},
            "real_aqi": {"$ne": None}
        },
        {"_id": 0, "timestamp": 1, "real_aqi": 1}
    ))

    if not data:
        return {
            "date": start_of_today.date().isoformat(),
            "avg_aqi": None,
            "message": "No AQI data available for today yet"
        }

    df = pd.DataFrame(data)
    df["real_aqi"] = pd.to_numeric(df["real_aqi"], errors="coerce")

    avg_aqi = round(df["real_aqi"].mean(), 2)

    return {
        "date": start_of_today.date().isoformat(),
        "avg_aqi": avg_aqi,
        "hours_recorded": len(df)
    }
