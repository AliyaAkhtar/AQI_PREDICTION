import numpy as np
from features.aqi_calculator import compute_overall_aqi

def add_time_features(df):
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.weekday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df

def add_cyclical_time_features(df):
    """Helps model understand cyclic nature of time"""
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df

def add_lag_features(df):
    """Past pollution levels strongly influence future AQI"""
    df = df.sort_values("timestamp")

    for lag in [1, 2, 3, 6, 12, 24, 48, 72]:
        df[f"pm2_5_lag_{lag}"] = df["pm2_5"].shift(lag)
        df[f"aqi_lag_{lag}"] = df["real_aqi"].shift(lag)

    return df

def add_rolling_features(df):
    """Rolling statistics capture pollution trends"""
    windows = [3, 6, 12, 24, 48]

    for w in windows:
        df[f"pm2_5_roll_mean_{w}"] = df["pm2_5"].rolling(w).mean()
        df[f"pm2_5_roll_std_{w}"] = df["pm2_5"].rolling(w).std()
        df[f"aqi_roll_mean_{w}"] = df["real_aqi"].shift(1).rolling(w).mean()

    return df

def add_weather_interactions(df):
    """Weather strongly affects pollution dispersion"""
    df["temp_x_pm25"] = df["temperature_2m"] * df["pm2_5"]
    df["wind_x_pm25"] = df["windspeed_10m"] * df["pm2_5"]
    df["humidity_x_pm25"] = df["relativehumidity_2m"] * df["pm2_5"]
    return df

def add_future_targets(df):
    """
    Create AQI targets for next 1, 2 and 3 days (24h intervals)
    """
    df = df.sort_values("timestamp")

    df["aqi_t_plus_24"] = df["real_aqi"].shift(-24)
    df["aqi_t_plus_48"] = df["real_aqi"].shift(-48)
    df["aqi_t_plus_72"] = df["real_aqi"].shift(-72)

    return df

def add_real_aqi(df):
    """
    Compute real AQI (0–500) from pollutant concentrations
    """
    df["real_aqi"] = df.apply(compute_overall_aqi, axis=1)
    return df
