import React, { useEffect, useState } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid, ResponsiveContainer,
  BarChart, Bar
} from "recharts";

const API_BASE = "http://127.0.0.1:8000";

export default function App() {
  const [forecast, setForecast] = useState([]);
  const [history, setHistory] = useState([]);
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    fetchForecast();
    fetchHistory();
    fetchMetrics();
  }, []);

  const fetchForecast = async () => {
    const res = await axios.get(`${API_BASE}/aqi/forecast`);
    setForecast(res.data.forecasts);
  };

  const fetchHistory = async () => {
    const res = await axios.get(`${API_BASE}/aqi/history?days=7`);
    setHistory(res.data.history);
  };

  const fetchMetrics = async () => {
    const res = await axios.get(`${API_BASE}/models/metrics/latest`);
    setMetrics(res.data);
  };

  return (
    <div style={{ padding: "30px", fontFamily: "Arial" }}>
      <h1>🌍 AQI Forecast Dashboard</h1>

      {/* ================= FORECAST ================= */}
      <h2>📅 3-Day AQI Forecast</h2>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={forecast}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="avg_aqi" name="Predicted AQI" />
        </BarChart>
      </ResponsiveContainer>

      {/* ================= HISTORY ================= */}
      <h2>📈 Historical AQI Trend</h2>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={history}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="aqi" name="Actual AQI" />
        </LineChart>
      </ResponsiveContainer>

      {/* ================= MODEL METRICS ================= */}
      <h2>🤖 Model Performance (Today)</h2>
      {metrics && (
        <div style={{ display: "flex", gap: "40px", flexWrap: "wrap" }}>
          
          {/* Production Model */}
          <div style={cardStyle}>
            <h3>🏆 Production Model</h3>
            <p><b>Name:</b> {metrics.production_model?.run_name}</p>
            <p><b>Version:</b> {metrics.production_model?.version}</p>
            <p>RMSE 24h: {metrics.production_model?.rmse_24h?.toFixed(2)}</p>
            <p>RMSE 48h: {metrics.production_model?.rmse_48h?.toFixed(2)}</p>
            <p>RMSE 72h: {metrics.production_model?.rmse_72h?.toFixed(2)}</p>
          </div>

          {/* Other Models */}
          {metrics.other_models?.map((m) => (
            <div key={m.version} style={cardStyle}>
              <h3>Model v{m.version}</h3>
              <p><b>Name:</b> {m.run_name}</p>
              <p>RMSE 24h: {m.rmse_24h?.toFixed(2)}</p>
              <p>RMSE 48h: {m.rmse_48h?.toFixed(2)}</p>
              <p>RMSE 72h: {m.rmse_72h?.toFixed(2)}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const cardStyle = {
  background: "#f5f5f5",
  padding: "15px 20px",
  borderRadius: "10px",
  boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
  minWidth: "250px"
};
