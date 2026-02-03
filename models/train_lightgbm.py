import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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

    base_model = lgb.LGBMRegressor(**params)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("regressor", MultiOutputRegressor(base_model))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    version, rmse = log_model(model, "LightGBM_AQI_Forecast", params, X_train, y_test, preds)
    return version, rmse
