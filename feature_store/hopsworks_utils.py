# import hopsworks
# import os
# from dotenv import load_dotenv

# load_dotenv()

# def get_feature_group():
#     project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
#     fs = project.get_feature_store()

#     fg = fs.get_or_create_feature_group(
#         name="aqi_weather_features",
#         version=1,
#         primary_key=["city", "timestamp"],
#         description="AQI + weather features",
#         online_enabled=False
#     )
#     return fg

# def get_lag_feature_group():
#     project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
#     fs = project.get_feature_store()

#     return fs.get_or_create_feature_group(
#     name="aqi_weather_lag_features",
#     version=1,
#     primary_key=["city", "timestamp"],
#     description="AQI Weather lag features for training"
# )


import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()

def get_feature_group():
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()

    from hsfs.feature import Feature
    
    features = [
        Feature(name="city", type="string"),
        Feature(name="timestamp", type="timestamp"),
        Feature(name="temperature_2m", type="double"),
        Feature(name="relativehumidity_2m", type="bigint"),
        Feature(name="pressure_msl", type="double"),
        Feature(name="windspeed_10m", type="double"),
        Feature(name="pm2_5", type="double"),
        Feature(name="pm10", type="double"),
        Feature(name="no2", type="double"),
        Feature(name="so2", type="double"),
        Feature(name="o3", type="double"),
        Feature(name="co", type="double"),
        Feature(name="us_aqi", type="bigint"),
        Feature(name="hour", type="int"),
        Feature(name="day_of_week", type="int"),
        Feature(name="is_weekend", type="int")
    ]

    fg = fs.get_or_create_feature_group(
        name="aqi_weather_features",
        version=2,
        primary_key=["city", "timestamp"],
        event_time="timestamp",  
        description="AQI + weather features",
        online_enabled=False,
        features=features  
    )
    return fg