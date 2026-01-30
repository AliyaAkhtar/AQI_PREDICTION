from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from feature_store.mongodb_store import load_features

def train_model(prepare_data, log_model):
    df = load_features()
    X_train, X_test, y_train, y_test = prepare_data(df)

    params = {
        "n_estimators": 200,
        "max_depth": 12,
        "random_state": 42
    }

    model = MultiOutputRegressor(RandomForestRegressor(**params))
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    log_model(model, "RandomForest_AQI_Forecast", params, y_test, preds)
