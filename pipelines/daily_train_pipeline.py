import subprocess
import sys
import os
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import models.train_random_forest as rf
import models.train_lightgbm as lgbm
import models.train_xgboost as xgb
import models.train_linear as lr
import os
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import mlflow
from dotenv import load_dotenv
from features.feature_engineering import add_future_targets

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

print("MLflow Tracking URI:", mlflow.get_tracking_uri())

TARGET_COLS = ["aqi_t_plus_24", "aqi_t_plus_48", "aqi_t_plus_72"]

def prepare_data(df):
    df = df.sort_values("timestamp")

    if "aqi_t_plus_24" not in df.columns:
        df = add_future_targets(df)

    df = df.dropna(subset=TARGET_COLS)

    drop_cols = [
        "timestamp",
        "city",
        "us_aqi",   
        "aqi_t_plus_24",
        "aqi_t_plus_48",
        "aqi_t_plus_72"
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[TARGET_COLS]

    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    return X_train, X_test, y_train, y_test

def log_model(model, run_name, params, X_train, y_test, preds):
    horizons = ["24h", "48h", "72h"]
    rmses = []

    with mlflow.start_run(run_name=run_name) as run:

        for k, v in params.items():
            mlflow.log_param(k, v)

        for i, h in enumerate(horizons):
            mae = mean_absolute_error(y_test.iloc[:, i], preds[:, i])
            rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], preds[:, i]))
            mlflow.log_metric(f"MAE_{h}", mae)
            mlflow.log_metric(f"RMSE_{h}", rmse)
            rmses.append(rmse)

        avg_rmse = float(np.mean(rmses))
        mlflow.log_metric("RMSE_avg", avg_rmse)

        signature = infer_signature(X_train, model.predict(X_train))

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_train.head(1)
        )

        mv = mlflow.register_model(
            model_uri=model_info.model_uri,
            name="AQI_Forecast_Model"
        )

        print(f" Registered {run_name} as version {mv.version} (AVG_RMSE={avg_rmse:.4f})")

        return int(mv.version), avg_rmse

def promote_best_of_today(versions_this_run):
    client = MlflowClient()
    model_name = "AQI_Forecast_Model"

    # pick lowest RMSE from today’s models only
    best_version, best_rmse = min(versions_this_run, key=lambda x: x[1])

    client.transition_model_version_stage(
        name=model_name,
        version=str(best_version),
        stage="Production",
        archive_existing_versions=True
    )

    print(f"\n BEST MODEL SELECTED")
    print(f"   ➜ Version: {best_version}")
    print(f"   ➜ Avg RMSE: {best_rmse:.4f}")
    print("   ➜ Promoted to PRODUCTION\n")

# PIPELINE: RUN ALL MODELS
versions_this_run = []

v, rmse = rf.train_model(prepare_data, log_model)
versions_this_run.append((v, rmse))

v, rmse = lgbm.train_model(prepare_data, log_model)
versions_this_run.append((v, rmse))

v, rmse = xgb.train_model(prepare_data, log_model)
versions_this_run.append((v, rmse))

v, rmse = lr.train_model(prepare_data, log_model)
versions_this_run.append((v, rmse))

# promote_best_model()
promote_best_of_today(versions_this_run)
