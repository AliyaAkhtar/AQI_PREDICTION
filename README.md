# 🌍 AQI Prediction System

A comprehensive **Air Quality Index (AQI) Prediction System** for real-time monitoring, forecasting, and visualization of air quality in cities. This repository integrates **data collection, feature engineering, machine learning modeling, and an interactive dashboard** for both historical and predicted AQI values.

---

## 🔹 Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Folder Structure](#folder-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)

---

## 🔹 Overview

Air pollution is a major public health concern, and timely AQI information is crucial for planning outdoor activities and policymaking. This repository provides:

1. **Historical AQI data tracking** — Collects and stores hourly AQI data.  
2. **AQI Forecasting** — Predicts air quality for the next 3–7 days using machine learning models.  
3. **Interactive Dashboard** — Displays historical data and forecasts in an easy-to-understand interface.  
4. **Model Performance Monitoring** — Tracks MAE, RMSE, and versioning for production and experimental models.

The system is designed to support **real-time AQI monitoring, forecasting, and visualization** for urban areas.

---

## 🔹 System Architecture

The AQI Prediction System is built using a **modular architecture**:

1. **Data Collection Layer**  
   - Collects real-time AQI data from external APIs or local sensors.  
   - Stores historical data for modeling and analysis.

2. **Feature Engineering Layer**  
   - Generates lag features and relevant environmental features for predictive modeling.

3. **Machine Learning Layer**  
   - Trains regression models (e.g., Random Forest, Gradient Boosting) to predict AQI for future hours/days.  
   - Tracks model metrics in **MLflow** for reproducibility.

4. **API Layer**  
   - FastAPI backend exposes endpoints to fetch historical data, forecasts, and model metrics.

5. **Frontend Dashboard**  
   - React-based dashboard that visualizes AQI trends interactively.  
   - Supports dynamic time range selection and chart types (Line/Area).  

---

## 🔹 Features

- Real-time AQI monitoring  
- Historical AQI analytics (hourly data)  
- 3-day AQI forecast with full day names and formatted dates  
- Color-coded AQI levels based on **EPA standards**  
- Interactive line/area charts  
- Production model tracking and evaluation metrics (MAE/RMSE)  
- Historical AQI visualization in an interactive graph  

---

## 🔹 Tech Stack

- **Backend**: Python, FastAPI  
- **Frontend**: React, Recharts, Tailwind CSS  
- **Database**: MongoDB (for historical and feature data)  
- **Machine Learning**: scikit-learn, pandas, numpy  
- **Model Tracking**: MLflow  
- **Environment**: Docker optional, Python virtual environment  

---

## 🔹 Folder Structure
.
├── api/                       # Backend FastAPI code, routes, controllers
├── aqi-dashboard-frontend/    # React frontend for AQI visualization
├── config/                    # Configuration files (environment variables, settings)
├── data_sources/              # Raw AQI or environmental datasets
├── feature_store/             # Generated features for ML models
├── features/                  # Scripts to build lag and derived features
├── models/                    # Trained ML models and saved artifacts
├── pipelines/                 # Automated ML pipelines for training and evaluation
├── .github/workflows/         # CI/CD pipelines for deployment or model training
├── .gitignore
├── mlflow.db                  # MLflow database for model tracking
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

**Key Notes:**
- `api/` exposes endpoints like `/aqi/history` and `/aqi/forecast`.
- `feature_store/` stores processed features ready for model input.
- `models/` contains production-ready and experimental model versions.
- `aqi-dashboard-frontend/` consumes API endpoints and renders charts & cards.

---

## 🔹 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/AliyaAkhtar/AQI_PREDICTION.git
cd AQI_PREDICTION
````

### 2. Backend Setup
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 3. Run Backend API

```bash
uvicorn api.main:app --reload
```

### 4. Frontend Setup

```bash
cd aqi-dashboard-frontend
npm install
npm start
```

---

## 🔹 Usage

* Access dashboard at: `http://localhost:3000`
* Fetch **historical AQI** and visualize interactively.
* Monitor **production model performance metrics**.

---

## 🔹 API Endpoints

| Endpoint                 | Method | Description                            |
| ------------------------ | ------ | -------------------------------------- |
| `/aqi/history?days=4`    | GET    | Fetch historical AQI for last 4 days   |
| `/aqi/forecast`          | GET    | Fetch 3-day AQI forecast               |
| `/models/metrics/latest` | GET    | Fetch production & other model metrics |

* Dashboard automatically fetches historical AQI and updates the interactive graph.
* AQI values are color-coded based on **EPA standards**.