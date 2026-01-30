def clean_data(df):
    df = df.sort_values("timestamp")

    # Remove impossible negative pollution values
    pollutant_cols = ["pm2_5", "pm10", "no2", "so2", "o3", "co"]
    for col in pollutant_cols:
        if col in df.columns:
            df.loc[df[col] < 0, col] = None

    # Forward & backward fill for small gaps
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df


def cap_outliers(df):
    """Cap extreme pollution spikes to reduce model noise"""
    cols = ["pm2_5", "pm10", "no2", "o3", "us_aqi"]
    for col in cols:
        if col in df.columns:
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=upper)
    return df