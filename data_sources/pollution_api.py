import requests
import pandas as pd
from datetime import datetime, timezone
from config.config import LAT, LON, OPENWEATHER_API_KEY


def fetch_pollution_history(start_dt, end_dt):
    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"

    params = {
        "lat": LAT,
        "lon": LON,
        "start": int(start_dt.timestamp()),
        "end": int(end_dt.timestamp()),
        "appid": OPENWEATHER_API_KEY
    }

    res = requests.get(url, params=params)
    res.raise_for_status()

    rows = []
    for item in res.json()["list"]:
        ts = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
        comp = item["components"]

        rows.append({
            "timestamp": ts,
            "pm2_5": comp.get("pm2_5"),
            "pm10": comp.get("pm10"),
            "no2": comp.get("no2"),
            "so2": comp.get("so2"),
            "o3": comp.get("o3"),
            "co": comp.get("co"),
            "us_aqi": item.get("main", {}).get("aqi")
        })

    return pd.DataFrame(rows)