import pandas as pd
from pymongo import MongoClient
import certifi
from config.config import MONGO_URI, MONGO_DB, MONGO_COLLECTION

def fetch_features(city="Karachi"):
    client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
    collection = client[MONGO_DB][MONGO_COLLECTION]

    data = list(collection.find({"city": city}))
    df = pd.DataFrame(data).drop(columns=["_id"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    return df