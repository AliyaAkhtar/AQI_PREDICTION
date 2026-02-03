import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from feature_store.mongodb_store import load_features

def train_model(prepare_data, log_model):
    print("Training Ridge model...")

    df = load_features()
    X_train, X_test, y_train, y_test = prepare_data(df)

    params = {"alpha": 1.0}

    base_model = Ridge(**params)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("regressor", MultiOutputRegressor(base_model))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    version, rmse = log_model(model, "Ridge_AQI_Forecast", params, X_train, y_test, preds)
    return version, rmse
