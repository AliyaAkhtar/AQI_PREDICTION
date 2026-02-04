from fastapi import APIRouter, HTTPException
from api.services.mongo_service import get_forecasts, get_history
from config.config import CITY

router = APIRouter(prefix="/aqi", tags=["AQI"])

@router.get("/forecast")
def aqi_forecast():
    data = get_forecasts()
    if not data:
        raise HTTPException(status_code=404, detail="Forecast not available yet.")
    return {"city": CITY, "forecasts": data}

@router.get("/history")
def aqi_history(days: int = 4):
    data = get_history(days)
    return {"history": data}
