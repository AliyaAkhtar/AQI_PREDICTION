import React, { useEffect, useState } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from "recharts";
import "./App.css";

const API_BASE = "http://localhost:8000"; // change when deployed

function App() {
  const [metrics, setMetrics] = useState([]);
  const [production, setProduction] = useState(null);
  const [chartData, setChartData] = useState([]);

  // 🔹 Load model metrics on page load
  useEffect(() => {
    axios.get(`${API_BASE}/models/metrics/latest`)
      .then(res => {
        setProduction(res.data.production_model);
        setMetrics(res.data.other_models || []);
      })
      .catch(err => console.error("Metrics error:", err));
  }, []);

  // 🔹 Fetch forecast + history when button clicked
  const loadAQIData = async () => {
    try {
      const forecastRes = await axios.get(`${API_BASE}/aqi/forecast`);
      const historyRes = await axios.get(`${API_BASE}/aqi/history?days=4`);

      const history = historyRes.data.history.map(item => ({
        date: item.timestamp,
        AQI: item.aqi
      }));

      const forecast = forecastRes.data.forecasts.map(item => ({
        date: item.timestamp,
        Forecast: item.predicted_aqi
      }));

      setChartData([...history, ...forecast]);
    } catch (err) {
      console.error("AQI fetch error:", err);
    }
  };

  const getAQIColor = (aqi) => {
    if (aqi <= 50) return "green";
    if (aqi <= 100) return "yellow";
    if (aqi <= 150) return "orange";
    if (aqi <= 200) return "red";
    if (aqi <= 300) return "purple";
    return "maroon";
  };

  return (
    <div className="container">
      <h1 className="title">🌍 AQI Forecast Dashboard</h1>

      {/* EPA AQI Info */}
      <div className="card info">
        <h2>EPA AQI Levels</h2>
        <ul>
          <li style={{color:"green"}}>Good (0-50)</li>
          <li style={{color:"gold"}}>Moderate (51-100)</li>
          <li style={{color:"orange"}}>Unhealthy for Sensitive Groups (101-150)</li>
          <li style={{color:"red"}}>Unhealthy (151-200)</li>
          <li style={{color:"purple"}}>Very Unhealthy (201-300)</li>
          <li style={{color:"maroon"}}>Hazardous (301+)</li>
        </ul>
      </div>

      {/* Model Metrics */}
      <div className="card">
        <h2>📊 Today's Model Performance</h2>
        <table>
          <thead>
            <tr>
              <th>Version</th>
              <th>Run Name</th>
              <th>RMSE 24h</th>
              <th>RMSE 48h</th>
              <th>RMSE 72h</th>
              <th>Stage</th>
            </tr>
          </thead>
          <tbody>
            {production && (
              <tr className="production">
                <td>{production.version}</td>
                <td>{production.run_name}</td>
                <td>{production.rmse_24h?.toFixed(2)}</td>
                <td>{production.rmse_48h?.toFixed(2)}</td>
                <td>{production.rmse_72h?.toFixed(2)}</td>
                <td>{production.stage}</td>
              </tr>
            )}
            {metrics.map((m, i) => (
              <tr key={i}>
                <td>{m.version}</td>
                <td>{m.run_name}</td>
                <td>{m.rmse_24h?.toFixed(2)}</td>
                <td>{m.rmse_48h?.toFixed(2)}</td>
                <td>{m.rmse_72h?.toFixed(2)}</td>
                <td>{m.stage}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* AQI Graph */}
      <div className="card">
        <h2>📈 AQI History & Forecast</h2>
        <button className="load-btn" onClick={loadAQIData}>
          Load AQI Data
        </button>

        {chartData.length > 0 && (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="AQI" stroke="#1f77b4" strokeWidth={3} />
              <Line type="monotone" dataKey="Forecast" stroke="#ff7300" strokeWidth={3} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}

export default App;
