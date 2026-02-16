# 🌍 AQI Prediction System

A comprehensive **Air Quality Index (AQI) Prediction System** for real-time monitoring, forecasting, and visualization of air quality in Karachi. This repository integrates **data collection, feature engineering, machine learning modeling, and an interactive dashboard** for both historical and predicted AQI values.

## 🔹 Table of Contents
- [Overview](#overview)
- [Live Demo](#livedemo)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Folder Structure](#folder-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)

## 🔹 Overview

Air pollution is a major public health concern, and timely AQI information is crucial for planning outdoor activities and policymaking. This repository provides:

1. **Historical AQI data tracking** — Collects and stores hourly AQI data.  
2. **AQI Forecasting** — Predicts air quality for the next 3 days using machine learning models.  
3. **Interactive Dashboard** — Displays historical data and forecasts in an easy-to-understand interface.  
4. **Model Performance Monitoring** — Tracks MAE, RMSE, and versioning for production and experimental models.

The system is designed to support **real-time AQI monitoring, forecasting, and visualization** for Karachi area.

## 🔹 Live Demo

AQI Prediction App Deployed:
👉 https://aqiprediction-ztsbvrbcmzttrd8qbrsx4u.streamlit.app/

## 🔹 System Architecture

The AQI Prediction System is built using a **modular architecture**:

1. **Data Collection Layer**  
   - Collects real-time AQI data from external APIs like OpenWeatherMap and OpenMeteo.  
   - Stores historical data for modeling and analysis using Mongodb as a feature store .

2. **Feature Engineering Layer**  
   - Generates lag features and relevant environmental features for predictive modeling.

3. **Machine Learning Layer**  
   - Trains regression models (Random Forest, XGBoost, Ridge & LightGBM) to predict AQI for future 3 days.  
   - Tracks model metrics in **MLflow** for reproducibility.

4. **Frontend Dashboard**  
   - Streamlit dashboard that visualizes AQI trends interactively.  
   - Displays the different metrics of models for comparison 

## 🔹 Features

- Real-time AQI monitoring  
- Historical AQI analytics (hourly data)  
- 3-day AQI forecast with full day names and formatted dates  
- Color-coded AQI levels based on **EPA standards**  
- Production model tracking and evaluation metrics (MAE/RMSE)  
- Historical AQI visualization in an interactive graph  

## 🔹 Tech Stack

- **Frontend**: Streamlit 
- **Feature Store**: MongoDB (for historical and feature data)  
- **Machine Learning**: scikit-learn, pandas, numpy  
- **Model Tracking**: MLflow, DagsHub  
- **Environment**: Python virtual environment  

## 🔹 Folder Structure

```text
.
├── streamlit_app/             # Frontend for AQI visualization
├── config/                    # Configuration files for settings
├── data_sources/              # Raw AQI or environmental data using different apis
├── feature_store/             # Connecting Mongodb as feature store
├── features/                  # Scripts to build lag and derived features
├── models/                    # Train ML models and saved artifacts
├── pipelines/                 # Automated ML pipelines for training and evaluation
├── notebooks& shap results/   # Shows the EDA and SHAP analysis results
├── .github/workflows/         # CI/CD pipelines for deployment or model training
├── .gitignore
├── mlflow.db                  # MLflow database for model tracking
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
````

**Key Notes:**

* `feature_store/` stores processed features ready for model input.
* `models/` contains production-ready and experimental model versions.
* `streamlit_app/` renders charts & cards.

## 🔹 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/AliyaAkhtar/AQI_PREDICTION.git
cd AQI_PREDICTION
```

### 2. Create virtual environment

```bash
python -m venv venv
# Activate virtual environment
# Linux / Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### 3. App Setup

```bash
cd streamlit_app
streamlit run app.py
```

## 🔹 Usage

* Access deployed dashboard at: `https://aqiprediction-ztsbvrbcmzttrd8qbrsx4u.streamlit.app/`
* Historical AQI is fetched automatically on load.
* Interactive graph shows **historical AQI** and **3-day forecast**.
* Monitor **production model performance metrics** in the dashboard table.
