from pymongo import MongoClient
import pandas as pd
from pymongo import MongoClient, UpdateOne
from config.config import MONGO_URI, MONGO_DB, MONGO_COLLECTION

client = MongoClient(MONGO_URI)
collection = client[MONGO_DB][MONGO_COLLECTION]

collection.create_index([("city", 1), ("timestamp", 1)], unique=True)

def upsert_features(df):
    ops = []
    for r in df.to_dict("records"):
        ops.append(
            UpdateOne(
                {"city": r["city"], "timestamp": r["timestamp"]},
                {"$set": r},
                upsert=True
            )
        )
    if ops:
        res = collection.bulk_write(ops)
        print(f"Inserted: {res.upserted_count}, Updated: {res.modified_count}")

def load_features(city=None):
    """
    If city is provided → used for prediction
    If city is None → load full historical dataset for training
    """

    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]

    if city:
        data = list(collection.find({"city": city}))
    else:
        data = list(collection.find())  

    df = pd.DataFrame(data)

    if "_id" in df.columns:
        df.drop(columns=["_id"], inplace=True)

    # print("Columns in DF:", df.columns.tolist())
    # print("First row:\n", df.head(1))
    # print("Number of rows:", len(df))

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def load_recent_history(hours=72, city=None):
    """
    Load recent historical rows from MongoDB to compute lag/rolling features.
    Default = last 72 hours (needed for 3-day lags)
    """
    from datetime import datetime, timedelta, timezone

    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]

    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

    query = {"timestamp": {"$gte": cutoff_time}}
    if city:
        query["city"] = city

    data = list(collection.find(query))

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    if "_id" in df.columns:
        df.drop(columns=["_id"], inplace=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


