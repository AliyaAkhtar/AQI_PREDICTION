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

# Use MLflow server locally
# mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

# TRAIN BASE FUNCTIONS
TARGET_COLS = ["aqi_t_plus_24", "aqi_t_plus_48", "aqi_t_plus_72"]

def prepare_data(df):
    """
    Prepare features (X) and multi-horizon targets (y)
    """

    df = df.dropna(subset=TARGET_COLS)

    drop_cols = [
        "timestamp",
        "city",
        "us_aqi",
        "aqi_t_plus_24",
        "aqi_t_plus_48",
        "aqi_t_plus_72"
    ]

    X = df.drop(columns=drop_cols)
    y = df[TARGET_COLS]

    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def log_model(model, run_name, params, y_test, preds):
    """
    Log multi-output metrics to MLflow
    """
    horizons = ["24h", "48h", "72h"]
    with mlflow.start_run(run_name=run_name):
        for k, v in params.items():
            mlflow.log_param(k, v)
        for i, h in enumerate(horizons):
            mae = mean_absolute_error(y_test.iloc[:, i], preds[:, i])
            rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], preds[:, i]))
            mlflow.log_metric(f"MAE_{h}", mae)
            mlflow.log_metric(f"RMSE_{h}", rmse)
        mlflow.sklearn.log_model(model, "model")
    print(f"{run_name} logged to MLflow")


# PIPELINE: RUN ALL MODELS
rf.train_model(prepare_data, log_model)
lgbm.train_model(prepare_data, log_model)
xgb.train_model(prepare_data, log_model)
lr.train_model(prepare_data, log_model)

# model_scripts = [
#     "models/train_random_forest.py",
#     "models/train_lightgbm.py",
#     "models/train_xgboost.py",
#     "models/train_linear.py"
# ]

# for script in model_scripts:
#     print(f"Running {script} ...")
#     # Run each model script as a separate Python process
#     module_name = script.replace("/", ".").replace(".py", "")
#     result = subprocess.run([sys.executable, "-m", module_name], capture_output=True, text=True)
    
#     # Print stdout and stderr
#     print(result.stdout)
#     if result.stderr:
#         print("Errors:", result.stderr)
    
#     print(f"{script} completed.\n")
