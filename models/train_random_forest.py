from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from feature_store.mongodb_store import load_features

def train_model(prepare_data, log_model):
    print("Training Random Forest model...")

    df = load_features()
    X_train, X_test, y_train, y_test = prepare_data(df)

    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }

    model = MultiOutputRegressor(XGBRegressor(**params))
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
   
    version, rmse = log_model(model, "RandomForest_AQI_Forecast", params, X_train, y_test, preds)
    return version, rmse
