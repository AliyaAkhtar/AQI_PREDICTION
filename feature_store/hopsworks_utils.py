import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()

def get_feature_group():
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_weather_features",
        version=1,
        primary_key=["city", "timestamp"],
        online_enabled=False
    )

    return fg


# def get_feature_store():
#     project = hopsworks.login(
#         api_key_value=os.getenv("HOPSWORKS_API_KEY")
#     )
#     return project.get_feature_store()

