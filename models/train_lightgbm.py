import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from feature_store.mongodb_store import load_features

def train_model(prepare_data, log_model):
    print("Training LightGBM model...")

    df = load_features()
    X_train, X_test, y_train, y_test = prepare_data(df)

    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": 42
    }

    model = MultiOutputRegressor(lgb.LGBMRegressor(**params))
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    log_model(model, "LightGBM_AQI_Forecast", params, y_test, preds)
