import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
from mlflow.tracking import MlflowClient
import os
from dotenv import load_dotenv

load_dotenv()

# ==================== CONFIGURATION ====================
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MODEL_NAME = os.getenv("MODEL_NAME", "AQI_Forecast_Model")
CITY = os.getenv("CITY", "Karachi")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

# MLflow setup
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

# ==================== DATABASE CONNECTIONS ====================
@st.cache_resource
def get_mongo_client():
    return MongoClient(MONGO_URI)

@st.cache_resource
def get_mlflow_client():
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()

# ==================== DATA FETCHING FUNCTIONS ====================
@st.cache_data(ttl=60)  # Cache for 1 minute to see updates faster
def get_forecasts():
    """Get future AQI forecasts - deduplicates by taking the latest prediction per date"""
    client = get_mongo_client()
    db = client[MONGO_DB]
    preds_col = db["aqi_forecasts_daily"]
    
    tomorrow = pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1)
    
    # Get all forecasts from tomorrow onwards
    data = list(preds_col.find({"date": {"$gte": tomorrow}}, {"_id": 0, "date": 1, "avg_aqi": 1}))
    
    if not data:
        return []
    
    # Convert to DataFrame and deduplicate by date (keep last entry for each date)
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Group by date and take the mean of avg_aqi to handle duplicates
    df = df.groupby('date').agg({'avg_aqi': 'mean'}).reset_index()
    df = df.sort_values('date')
    
    return df.to_dict('records')

@st.cache_data(ttl=60)
def get_history(days: int):
    """Get historical AQI data"""
    client = get_mongo_client()
    db = client[MONGO_DB]
    features_col = db["features_karachi_hourly"]
    
    end = pd.Timestamp.utcnow().replace(tzinfo=timezone.utc)
    start = end - timedelta(days=days)
    
    data = list(features_col.find(
        {"timestamp": {"$gte": start}},
        {"_id": 0, "timestamp": 1, "real_aqi": 1}
    ))
    
    if not data:
        return {"history": []}
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.date
    
    # Calculate daily average
    daily_avg = (
        df.groupby("date")["real_aqi"]
        .mean()
        .reset_index()
    )
    
    daily_avg["real_aqi"] = daily_avg["real_aqi"].round(2)
    return {"history": daily_avg.to_dict(orient="records")}

@st.cache_data(ttl=60)
def get_today_avg_aqi():
    """Get today's average AQI"""
    client = get_mongo_client()
    db = client[MONGO_DB]
    features_col = db["features_karachi_hourly"]
    
    now = pd.Timestamp.utcnow().replace(tzinfo=timezone.utc)
    start_of_today = now.normalize()
    end_of_today = start_of_today + pd.Timedelta(days=1)
    
    data = list(features_col.find(
        {
            "timestamp": {"$gte": start_of_today, "$lt": end_of_today},
            "real_aqi": {"$ne": None}
        },
        {"_id": 0, "timestamp": 1, "real_aqi": 1}
    ))
    
    if not data:
        return {
            "date": start_of_today.date().isoformat(),
            "avg_aqi": None,
            "message": "No AQI data available for today yet"
        }
    
    df = pd.DataFrame(data)
    df["real_aqi"] = pd.to_numeric(df["real_aqi"], errors="coerce")
    avg_aqi = round(df["real_aqi"].mean(), 2)
    
    return {
        "date": start_of_today.date().isoformat(),
        "avg_aqi": avg_aqi,
        "hours_recorded": len(df)
    }

@st.cache_data(ttl=600)
def fetch_model_metrics():
    client = get_mlflow_client()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    today = datetime.now(timezone.utc).date()
    
    production = None
    others = []
    
    for v in versions:
        run = client.get_run(v.run_id)
        run_start = datetime.fromtimestamp(run.info.start_time / 1000, tz=timezone.utc).date()
        
        if run_start != today:
            continue
        
        metrics = run.data.metrics
        
        info = {
            "version": int(v.version),
            "run_name": run.data.tags.get("mlflow.runName"),
            "stage": v.current_stage,
            "mae_24h": metrics.get("MAE_24h"),
            "mae_48h": metrics.get("MAE_48h"),
            "mae_72h": metrics.get("MAE_72h"),
            "rmse_24h": metrics.get("RMSE_24h"),
            "rmse_48h": metrics.get("RMSE_48h"),
            "rmse_72h": metrics.get("RMSE_72h"),
            "rmse_avg": metrics.get("RMSE_avg"),
        }
        
        if v.current_stage == "Production":
            production = info
        else:
            others.append(info)
    
    return production, others

# ==================== UTILITY FUNCTIONS ====================
def get_aqi_color(aqi):
    if aqi is None:
        return "#94a3b8"
    if aqi <= 50:
        return "#10b981"
    if aqi <= 100:
        return "#fbbf24"
    if aqi <= 150:
        return "#f97316"
    if aqi <= 200:
        return "#ef4444"
    if aqi <= 300:
        return "#a855f7"
    return "#7f1d1d"

def get_aqi_label(aqi):
    if aqi is None:
        return "No Data"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"

def get_aqi_gradient(aqi):
    """Get gradient background for AQI value"""
    if aqi is None:
        return "linear-gradient(135deg, #475569 0%, #334155 100%)"
    if aqi <= 50:
        return "linear-gradient(135deg, #10b981 0%, #059669 100%)"
    if aqi <= 100:
        return "linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%)"
    if aqi <= 150:
        return "linear-gradient(135deg, #f97316 0%, #ea580c 100%)"
    if aqi <= 200:
        return "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)"
    if aqi <= 300:
        return "linear-gradient(135deg, #a855f7 0%, #9333ea 100%)"
    return "linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%)"

# ==================== STREAMLIT APP ====================
def main():
    st.set_page_config(
        page_title=f"AQI Forecast - {CITY}",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Enhanced Custom CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Force dark background */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #ffffff 100%) !important;
        color: #0f172a;
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f0f4f8 0%, #ffffff 100%) !important;
        color: #0f172a;
    }
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%) !important;
        background-attachment: fixed;
    }
    
    /* Animated Background */
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            radial-gradient(circle at 20% 30%, rgba(56, 189, 248, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 80% 70%, rgba(167, 139, 250, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 50% 50%, rgba(34, 211, 238, 0.1) 0%, transparent 50%);
        animation: backgroundPulse 15s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes backgroundPulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.1); }
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
        position: relative;
        z-index: 1;
    }
    
    /* Enhanced Headers */
    h1 {
        font-family: 'Poppins', sans-serif !important;
        color: #1e293b !important;
        font-size: 5rem !important;
        font-weight: 900 !important;
        text-align: center;
        letter-spacing: -0.02em !important;
        line-height: 1.1 !important;
        margin-bottom: 0.25rem !important;
        text-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
    }
    
    @keyframes titleGlow {
        0%, 100% { filter: drop-shadow(0 0 20px rgba(56, 189, 248, 0.4)); }
        50% { filter: drop-shadow(0 0 40px rgba(56, 189, 248, 0.7)); }
    }
    
    h3 {
        font-family: 'Poppins', sans-serif !important;
        color: #1e293b !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
        margin-top: 2.5rem !important;
        margin-bottom: 1.5rem !important;
        letter-spacing: -0.01em !important;
        text-align: left;
        position: relative;
        padding-left: 1rem;
    }
    
    h3::before {
        content: '';
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 6px;
        height: 70%;
        background: linear-gradient(180deg, #38bdf8 0%, #a78bfa 100%);
        border-radius: 3px;
    }
    
    /* Glassmorphic Cards */
    .aqi-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .aqi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--card-gradient, linear-gradient(90deg, #38bdf8, #a78bfa));
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .aqi-card:hover {
        transform: translateY(-8px);
        background: rgba(255, 255, 255, 0.05);
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }
    
    .aqi-card:hover::before {
        opacity: 1;
    }
    
    /* Primary AQI Card */
    .aqi-primary-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 32px;
        padding: 3rem;
        position: relative;
        overflow: hidden;
        transition: all 0.5s ease;
        animation: cardFloat 6s ease-in-out infinite;
    }
    
    @keyframes cardFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .aqi-primary-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(
            from 0deg,
            transparent,
            var(--glow-color, #38bdf8),
            transparent 30%
        );
        animation: rotate 8s linear infinite;
        opacity: 0.3;
    }
    
    @keyframes rotate {
        100% { transform: rotate(360deg); }
    }
    
    .aqi-primary-card::after {
        content: '';
        position: absolute;
        inset: 2px;
        background: linear-gradient(135deg, rgba(10, 14, 39, 0.95) 0%, rgba(26, 31, 58, 0.95) 100%);
        border-radius: 30px;
        z-index: 1;
    }
    
    .aqi-card-content {
        position: relative;
        z-index: 2;
    }
    
    .aqi-value {
        font-family: 'Poppins', sans-serif;
        font-size: 5rem;
        font-weight: 800;
        line-height: 1;
        margin: 1rem 0;
        text-shadow: 0 0 30px currentColor;
        animation: pulse 3s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.05); opacity: 0.9; }
    }
    
    .aqi-label-top {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .aqi-date {
        font-size: 1rem;
        color: #64748b;
        margin-top: 0.5rem;
    }
    
    .aqi-status {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1rem;
        margin-top: 1rem;
        letter-spacing: 1px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* =================== PRODUCTION MODEL BOX =================== */
    .production-box {
        background: linear-gradient(135deg, #38bdf8 0%, #6366f1 100%);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        color: #ffffff !important;
        font-weight: 600;
        box-shadow: 0 8px 24px rgba(56, 189, 248, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .production-box:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(56, 189, 248, 0.45);
    }
    .production-box span {
        font-weight: 700;
        color: #ffffff;
    }
                
    /* Info Cards */
    .info-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.12);
        transform: translateY(-4px);
    }
    
    /* Charts */
    .stPlotlyChart {
        background: transparent;
        border: none;
        padding: 0;
    }
    
    /* DataFrames */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0,0,0,0.05);
        border-radius: 20px;
        overflow: hidden;
    }
    
    /* Info Boxes */
    .stAlert {
        background: rgba(56, 189, 248, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 16px;
        color: #e2e8f0;
    }
    
    .caption {
        font-family: 'Inter', sans-serif !important;
        color: #475569;
        font-size: 1.5rem !important;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.5);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #38bdf8 0%, #3b82f6 100%);
        border-radius: 10px;
        border: 2px solid rgba(15, 23, 42, 0.5);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown(f'<h1>🌍 {CITY} Air Quality</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="caption">Real-time monitoring & ML-powered forecasting', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Fetch data
    try:
        today_data = get_today_avg_aqi()
        forecasts = get_forecasts()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return
    
    # Today's AQI + 3-Day Forecast Cards
    st.markdown(f'<h3>📊 Current Air Quality & 3-Day Forecast', unsafe_allow_html=True)
    cols = st.columns(4, gap="large")
    
    # Today's AQI
    with cols[0]:
        if today_data and today_data.get("avg_aqi"):
            aqi_val = today_data["avg_aqi"]
            date_obj = datetime.fromisoformat(today_data["date"])
            weekday = date_obj.strftime("%A")
            formatted_date = date_obj.strftime("%B %d, %Y")
            
            color = get_aqi_color(aqi_val)
            label = get_aqi_label(aqi_val)
            gradient = get_aqi_gradient(aqi_val)
            
            st.markdown(f"""
            <div class="aqi-primary-card" style="--glow-color: {color};">
                <div class="aqi-card-content">
                    <div class="aqi-label-top">{weekday}</div>
                    <div class="aqi-value" style="color: {color};">{int(aqi_val)}</div>
                    <div class="aqi-date">{formatted_date}</div>
                    <div class="aqi-status" style="background: {gradient}; color: white;">
                        {label}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Today's AQI data not yet available")
    
    # 3-Day Forecast Cards
    for i, forecast in enumerate(forecasts[:3]):
        with cols[i + 1]:
            date_obj = pd.to_datetime(forecast["date"])
            weekday = date_obj.strftime("%A")
            formatted_date = date_obj.strftime("%B %d, %Y")
            aqi_val = forecast["avg_aqi"]
            
            color = get_aqi_color(aqi_val)
            label = get_aqi_label(aqi_val)
            gradient = get_aqi_gradient(aqi_val)
            
            st.markdown(f"""
            <div class="aqi-card" style="--card-gradient: {gradient};">
                <div style="font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.75rem;">
                    {weekday}
                </div>
                <div style="font-family: 'Poppins', sans-serif; font-size: 3.5rem; font-weight: 800; color: {color}; line-height: 1; margin-bottom: 0.5rem;">
                    {int(aqi_val)}
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 1rem;">
                    {formatted_date}
                </div>
                <div style="display: inline-block; padding: 0.4rem 1rem; border-radius: 50px; font-weight: 600; font-size: 0.85rem; background: {gradient}; color: white;">
                    {label}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # EPA AQI Levels
    st.markdown(f'<h3>📋 EPA Air Quality Index Guide', unsafe_allow_html=True)
    aqi_levels = [
        {"range": "0-50", "label": "Good", "color": "#10b981", "desc": "Air quality is satisfactory, minimal health concern"},
        {"range": "51-100", "label": "Moderate", "color": "#fbbf24", "desc": "Acceptable quality, some pollutants may affect sensitive individuals"},
        {"range": "101-150", "label": "Unhealthy for Sensitive", "color": "#f97316", "desc": "Sensitive groups may experience health effects"},
        {"range": "151-200", "label": "Unhealthy", "color": "#ef4444", "desc": "Everyone may experience health effects"},
        {"range": "201-300", "label": "Very Unhealthy", "color": "#a855f7", "desc": "Health alert - serious health effects for everyone"},
        {"range": "301+", "label": "Hazardous", "color": "#7f1d1d", "desc": "Emergency conditions - entire population affected"}
    ]
    
    level_cols = st.columns(3, gap="medium")
    for i, level in enumerate(aqi_levels):
        with level_cols[i % 3]:
            st.markdown(f"""
            <div class="info-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <div style="width: 4px; height: 40px; background: {level['color']}; border-radius: 4px;"></div>
                        <span style="color: {level['color']}; font-weight: 700; font-size: 1.1rem;">{level['label']}</span>
                    </div>
                    <span style="color: #64748b; font-weight: 600; font-size: 0.9rem; background: rgba(255,255,255,0.05); padding: 0.25rem 0.75rem; border-radius: 20px;">{level['range']}</span>
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.5;">
                    {level['desc']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Model Performance
    st.markdown(f'<h3> 🤖 ML Model Performance Metrics', unsafe_allow_html=True)
    try:
        production, others = fetch_model_metrics()
        
        if production:
            st.markdown(f"""
            <div class="production-box">
                🚀 Forecasts powered by <span>{production['run_name']}</span> — 
                our production model with lowest error metrics
            </div>
            """, unsafe_allow_html=True)
            
            all_models = [production] + others
            df_models = pd.DataFrame(all_models)
            
            df_models = df_models[[
                'version', 'run_name', 'stage',
                'mae_24h', 'mae_48h', 'mae_72h',
                'rmse_24h', 'rmse_48h', 'rmse_72h', 'rmse_avg'
            ]]
            
            metric_cols = ['mae_24h', 'mae_48h', 'mae_72h', 'rmse_24h', 'rmse_48h', 'rmse_72h', 'rmse_avg']
            for col in metric_cols:
                df_models[col] = df_models[col].round(2)
            
            df_models.columns = [
                'Version', 'Model Name', 'Stage',
                'MAE 24h', 'MAE 48h', 'MAE 72h',
                'RMSE 24h', 'RMSE 48h', 'RMSE 72h', 'Avg RMSE'
            ]
            
            st.dataframe(
                df_models,
                use_container_width=True,
                hide_index=True,
                height=min(len(df_models) * 35 + 38, 400)
            )
        else:
            st.warning("⚠️ No production model metrics available for today")
    except Exception as e:
        st.error(f"Error loading model metrics: {e}")
    
    # 7-Day Chart (3 history + today + 3 forecast)
    st.markdown(f'<h3> 📈 7-Day AQI Trend (Historical & Forecast)', unsafe_allow_html=True)

    try:
        # Get 3 days of history
        history_data = get_history(3)
        history_df = pd.DataFrame(history_data["history"])
        
        if not history_df.empty:
            history_df["date"] = pd.to_datetime(history_df["date"])
            history_df["type"] = "Historical"
            history_df = history_df.rename(columns={"real_aqi": "AQI"})
        
        # Get forecast data (should have 3 future days)
        forecast_df = pd.DataFrame(forecasts[:3])  # Only take first 3 forecasts
        if not forecast_df.empty:
            forecast_df["date"] = pd.to_datetime(forecast_df["date"])
            forecast_df["type"] = "Forecast"
            forecast_df = forecast_df.rename(columns={"avg_aqi": "AQI"})
        
        # Add today's data
        today_df = pd.DataFrame([{
            "date": pd.to_datetime(today_data["date"]),
            "AQI": today_data.get("avg_aqi"),
            "type": "Today"
        }]) if today_data.get("avg_aqi") else pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat([history_df, today_df, forecast_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
        combined_df = combined_df.sort_values("date").reset_index(drop=True)
        
        # Create chart
        fig = go.Figure()
        
        # Single continuous line connecting all points
        fig.add_trace(go.Scatter(
            x=combined_df["date"],
            y=combined_df["AQI"],
            mode='lines+markers',
            name='AQI Trend',
            line=dict(color='#38bdf8', width=3, shape='spline'),
            marker=dict(size=8, color='#38bdf8', line=dict(width=2, color='#0c4a6e')),
            hovertemplate='<b>%{x|%a, %b %d}</b><br>AQI: <b>%{y:.1f}</b><extra></extra>',
        ))
        
        # Highlight today
        today_point = combined_df[combined_df["type"] == "Today"]
        if not today_point.empty:
            fig.add_trace(go.Scatter(
                x=today_point["date"],
                y=today_point["AQI"],
                mode='markers',
                name='Today',
                marker=dict(size=14, color='#10b981', symbol='circle', line=dict(width=3, color='#ffffff')),
                hovertemplate='<b>TODAY</b><br>%{x|%a, %b %d}<br>AQI: <b>%{y:.1f}</b><extra></extra>',
            ))
        
        # Highlight forecast points
        forecast_point = combined_df[combined_df["type"] == "Forecast"]
        if not forecast_point.empty:
            fig.add_trace(go.Scatter(
                x=forecast_point["date"],
                y=forecast_point["AQI"],
                mode='markers',
                name='Forecast',
                marker=dict(size=10, color='#a78bfa', symbol='diamond', line=dict(width=2, color='#5b21b6')),
                hovertemplate='<b>Forecast</b><br>%{x|%a, %b %d}<br>AQI: <b>%{y:.1f}</b><extra></extra>',
            ))
        
        # Layout
        fig.update_layout(
            height=500,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', family='Inter', size=13),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(148,163,184,0.08)',
                gridwidth=1,
                title=None,
                tickfont=dict(size=12, color='#94a3b8'),
                showline=True,
                linecolor='rgba(148, 163, 184, 0.2)',
                linewidth=2
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(148, 163, 184, 0.1)',
                gridwidth=1,
                title=dict(
                    text='AQI Value',
                    font=dict(size=14, color='#cbd5e1')
                ),
                tickfont=dict(size=12, color='#94a3b8'),
                showline=True,
                linecolor='rgba(148, 163, 184, 0.2)',
                linewidth=2
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(15, 23, 42, 0.8)',
                bordercolor='rgba(148, 163, 184, 0.2)',
                borderwidth=1,
                font=dict(size=13, color='#e2e8f0')
            ),
            margin=dict(l=60, r=40, t=40, b=60),
            hoverlabel=dict(
                bgcolor='rgba(15, 23, 42, 0.95)',
                font_size=13,
                font_family='Inter',
                bordercolor='#38bdf8'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error creating visualization: {e}")
        import traceback
        st.code(traceback.format_exc())
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 2rem; border-top: 1px solid rgba(148, 163, 184, 0.1);">
        <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 0.5rem;">
            Data refreshes every minute • Powered by MongoDB, MLflow & Streamlit
        </p>
        <p style="color: #475569; font-size: 0.85rem;">
            🌍 Real-time environmental monitoring for a healthier tomorrow
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()