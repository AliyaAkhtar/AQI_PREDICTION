import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
from feature_store.mongodb_store import load_features

def train_model(prepare_data, log_model):
    print("Training Ridge model...")

    # Load data
    df = load_features()
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Impute missing values in features
    imputer = SimpleImputer(strategy="mean")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Train Ridge
    params = {"alpha": 1.0}
    model = MultiOutputRegressor(Ridge(**params))
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)

    # Log to MLflow
    # log_model(model, "Ridge_AQI_Forecast", params, y_test, preds)
    # log_model(model, "Ridge_AQI_Forecast", params, X_train, y_test, preds)
    
    version, rmse = log_model(model, "Ridge_AQI_Forecast", params, X_train, y_test, preds)
    return version, rmse
