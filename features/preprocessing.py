def clean_data(df):
    df = df.sort_values("timestamp")

    pollutant_cols = ["pm2_5", "pm10", "no2", "so2", "o3", "co"]

    for col in pollutant_cols:
        if col in df.columns:
            df.loc[df[col] < 0, col] = None

            # Only fill small gaps (up to 3 hours)
            df[col] = df[col].ffill(limit=3)
            df[col] = df[col].bfill(limit=3)
            
    return df

def cap_outliers(df):
    """Cap extreme pollution spikes to reduce model noise"""
    cols = ["pm2_5", "pm10", "no2", "o3", "real_aqi"]
    for col in cols:
        if col in df.columns:
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=upper)
    return df