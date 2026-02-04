from fastapi import APIRouter
from api.services.mlflow_service import fetch_model_metrics

router = APIRouter(prefix="/models", tags=["Models"])

@router.get("/metrics/latest")
def latest_model_metrics():
    production, others = fetch_model_metrics()
    return {
        "production_model": production,
        "other_models": others
    }
