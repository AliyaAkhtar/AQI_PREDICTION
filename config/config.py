import os
from dotenv import load_dotenv

load_dotenv()

CITY = os.getenv("CITY", "Karachi")
LAT = float(os.getenv("LAT", 24.8607))
LON = float(os.getenv("LON", 67.0011))
BACKFILL_DAYS = int(os.getenv("BACKFILL_DAYS", 90))

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")

# MLflow
# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns") 

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")
