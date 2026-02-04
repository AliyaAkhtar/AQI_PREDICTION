from mlflow.tracking import MlflowClient
from config.config import MODEL_NAME
from datetime import datetime, timezone

client = MlflowClient()

def fetch_model_metrics():
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    today = datetime.now(timezone.utc).date()

    production = None
    others = []

    for v in versions:
        run = client.get_run(v.run_id)
        run_start = datetime.fromtimestamp(run.info.start_time / 1000, tz=timezone.utc).date()

        # Only models trained today
        if run_start != today:
            continue

        metrics = run.data.metrics

        info = {
            "version": int(v.version),
            "run_name": run.data.tags.get("mlflow.runName"),
            "stage": v.current_stage,

            # MAE
            "mae_24h": metrics.get("MAE_24h"),
            "mae_48h": metrics.get("MAE_48h"),
            "mae_72h": metrics.get("MAE_72h"),

            # RMSE
            "rmse_24h": metrics.get("RMSE_24h"),
            "rmse_48h": metrics.get("RMSE_48h"),
            "rmse_72h": metrics.get("RMSE_72h"),

            # Overall Score
            "rmse_avg": metrics.get("RMSE_avg"),
        }

        if v.current_stage == "Production":
            production = info
        else:
            others.append(info)

    return production, others
