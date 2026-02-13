from fastapi import FastAPI
from api.routers import models, aqi
import mlflow
from config.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_TRACKING_USERNAME,
    MLFLOW_TRACKING_PASSWORD
)
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="AQI Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MLflow Auth
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

app.include_router(models.router)
app.include_router(aqi.router)

@app.get("/")
def root():
    return {"message": "AQI Forecast API is running"}