import React, { useEffect, useState } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  AreaChart, Area
} from "recharts";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [metrics, setMetrics] = useState([]);
  const [production, setProduction] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedTimeRange, setSelectedTimeRange] = useState(4);
  const [viewMode, setViewMode] = useState("line");
  const [lastUpdate, setLastUpdate] = useState(null);
  const [metricsLoading, setMetricsLoading] = useState(true);
  const [threeDayForecast, setThreeDayForecast] = useState([]);

  // Load model metrics
  useEffect(() => {
    const loadMetrics = async () => {
      setMetricsLoading(true);
      try {
        const res = await axios.get(`${API_BASE}/models/metrics/latest`);
        setProduction(res.data.production_model);
        setMetrics(res.data.other_models || []);
        setLastUpdate(new Date());
      } catch (err) {
        console.error("Metrics error:", err);
      } finally {
        setMetricsLoading(false);
      }
    };
    loadMetrics();
  }, []);

  // Load 3-day forecast
  useEffect(() => {
    const loadThreeDayForecast = async () => {
      try {
        const res = await axios.get(`${API_BASE}/aqi/forecast`);
        const forecasts = res.data.forecasts.map(f => {
          const dateObj = new Date(f.date);
          return {
            ...f,
            weekday: dateObj.toLocaleDateString("en-US", { weekday: "long" }),
            formattedDate: dateObj.toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })
          };
        });
        setThreeDayForecast(forecasts);
      } catch (err) {
        console.error("Forecast fetch error:", err);
      }
    };
    loadThreeDayForecast();
  }, []);

  // Load historical AQI automatically
  useEffect(() => {
    const loadAQIData = async () => {
      setLoading(true);
      try {
        const historyRes = await axios.get(`${API_BASE}/aqi/history?days=${selectedTimeRange}`);
        const history = historyRes.data.history.map(item => ({
          date: new Date(item.timestamp).toLocaleString("en-US", {
            weekday: "short",
            year: "numeric",
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit"
          }),
          AQI: item.real_aqi
        }));
        setChartData(history);
        setLastUpdate(new Date());
      } catch (err) {
        console.error("AQI fetch error:", err);
      } finally {
        setLoading(false);
      }
    };
    loadAQIData();
  }, [selectedTimeRange]);

  const getAQIColor = (aqi) => {
    if (aqi <= 50) return "#10b981";
    if (aqi <= 100) return "#fbbf24";
    if (aqi <= 150) return "#f97316";
    if (aqi <= 200) return "#ef4444";
    if (aqi <= 300) return "#a855f7";
    return "#7f1d1d";
  };

  const getAQILabel = (aqi) => {
    if (aqi <= 50) return "Good";
    if (aqi <= 100) return "Moderate";
    if (aqi <= 150) return "Unhealthy for Sensitive Groups";
    if (aqi <= 200) return "Unhealthy";
    if (aqi <= 300) return "Very Unhealthy";
    return "Hazardous";
  };

  const getLevel = (aqi) => {
    if (aqi <= 50) return { label: "Good", color: "#10b981" };
    if (aqi <= 100) return { label: "Moderate", color: "#fbbf24" };
    if (aqi <= 150) return { label: "Unhealthy for Sensitive Groups", color: "#f97316" };
    if (aqi <= 200) return { label: "Unhealthy", color: "#ef4444" };
    if (aqi <= 300) return { label: "Very Unhealthy", color: "#a855f7" };
    return { label: "Hazardous", color: "#7f1d1d" };
  };

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const value = data.AQI || data.Forecast;

      return (
        <div style={{
          background: "rgba(0, 0, 0, 0.9)",
          padding: "12px",
          borderRadius: "8px",
          border: `2px solid ${getAQIColor(value)}`
        }}>
          <p style={{ margin: 0, fontWeight: "bold", color: "#fff" }}>{data.date}</p>
          <p style={{ margin: "4px 0", color: getAQIColor(value) }}>
            {data.AQI ? "AQI" : "Forecast"}: {value?.toFixed(1)}
          </p>
          <p style={{ margin: 0, fontSize: "12px", color: "#aaa" }}>
            {getAQILabel(value)}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="container">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <h1 className="title">🌍 AQI Forecast Dashboard</h1>
          <div className="header-controls">
            {lastUpdate && (
              <span className="last-update">
                Last updated: {lastUpdate.toLocaleTimeString()}
              </span>
            )}
          </div>
        </div>
      </header>

      {/* Stats Cards */}
      <div className="stats-grid">
        {/* Production Model */}
        <div className="stat-card" style={{ borderLeft: "4px solid #3b82f6" }}>
          <div className="stat-label">Production Model</div>
          <div className="stat-value" style={{ fontSize: "1.5rem" }}>
            {production ? production.run_name : "Loading..."}
          </div>
        </div>

        {/* 3-Day Forecast Cards */}
        {threeDayForecast.map((f, idx) => {
          const level = getLevel(f.avg_aqi);
          return (
            <div
              key={idx}
              className="stat-card"
              style={{ borderLeft: `4px solid ${level.color}` }}
            >
              <div className="stat-label">{f.weekday}</div>
              <div className="stat-value" style={{ color: level.color }}>
                {f.avg_aqi.toFixed(0)}
              </div>
              <div className="stat-sublabel">{f.formattedDate}</div>
              <div
                className="aqi-level"
                style={{ marginTop: "4px", fontWeight: "bold", color: level.color }}
              >
                {level.label}
              </div>
            </div>
          );
        })}
      </div>

      {/* EPA AQI Info */}
      <div className="card info-card">
        <h2>📋 EPA AQI Levels</h2>
        <div className="aqi-levels">
          {[
            { range: "0-50", label: "Good", color: "#10b981", desc: "Air quality is satisfactory" },
            { range: "51-100", label: "Moderate", color: "#fbbf24", desc: "Acceptable for most people" },
            { range: "101-150", label: "Unhealthy for Sensitive", color: "#f97316", desc: "Sensitive groups may experience effects" },
            { range: "151-200", label: "Unhealthy", color: "#ef4444", desc: "Everyone may begin to experience effects" },
            { range: "201-300", label: "Very Unhealthy", color: "#a855f7", desc: "Health alert" },
            { range: "301+", label: "Hazardous", color: "#7f1d1d", desc: "Emergency conditions" }
          ].map((level, i) => (
            <div key={i} className="aqi-level-item" style={{ borderLeft: `4px solid ${level.color}` }}>
              <div className="aqi-level-header">
                <span style={{ color: level.color, fontWeight: "bold" }}>{level.label}</span>
                <span className="aqi-range">{level.range}</span>
              </div>
              <div className="aqi-level-desc">{level.desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Model Performance */}
      <div className="card">
        <h2>📊 Model Performance Metrics</h2>
        {metricsLoading && (
          <div className="table-loader">
            ⏳ Fetching latest model metrics...
          </div>
        )}

        {!metricsLoading && (
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Version</th>
                  <th>Run Name</th>
                  <th>MAE 24h</th>
                  <th>MAE 48h</th>
                  <th>MAE 72h</th>
                  <th>RMSE 24h</th>
                  <th>RMSE 48h</th>
                  <th>RMSE 72h</th>
                  <th>Avg RMSE</th>
                  <th>Stage</th>
                </tr>
              </thead>
              <tbody>
                {production && (
                  <tr className="production-row">
                    <td>
                      <span className="badge badge-production">🏆 v{production.version}</span>
                    </td>
                    <td><strong>{production.run_name}</strong></td>
                    <td>{production.mae_24h?.toFixed(2)}</td>
                    <td>{production.mae_48h?.toFixed(2)}</td>
                    <td>{production.mae_72h?.toFixed(2)}</td>
                    <td>{production.rmse_24h?.toFixed(2)}</td>
                    <td>{production.rmse_48h?.toFixed(2)}</td>
                    <td>{production.rmse_72h?.toFixed(2)}</td>
                    <td><strong>{production.rmse_avg?.toFixed(2)}</strong></td>
                    <td><span className="badge badge-stage-production">🚀 Production</span></td>
                  </tr>
                )}

                {metrics.map((m, i) => (
                  <tr key={i}>
                    <td>v{m.version}</td>
                    <td>{m.run_name}</td>
                    <td>{m.mae_24h?.toFixed(2)}</td>
                    <td>{m.mae_48h?.toFixed(2)}</td>
                    <td>{m.mae_72h?.toFixed(2)}</td>
                    <td>{m.rmse_24h?.toFixed(2)}</td>
                    <td>{m.rmse_48h?.toFixed(2)}</td>
                    <td>{m.rmse_72h?.toFixed(2)}</td>
                    <td>{m.rmse_avg?.toFixed(2)}</td>
                    <td><span className="badge">{m.stage}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* AQI Chart */}
      <div className="card">
        <div className="chart-header">
          <h2>📈 AQI History</h2>
          <div className="chart-controls">
            <div className="button-group">
              <button
                className={`btn-toggle ${selectedTimeRange === 4 ? "active" : ""}`}
                onClick={() => {
                  setSelectedTimeRange(4);
                }}
              >
                4 Days
              </button>
              <button
                className={`btn-toggle ${selectedTimeRange === 7 ? "active" : ""}`}
                onClick={() => {
                  setSelectedTimeRange(7);
                }}
              >
                7 Days
              </button>
            </div>
            <div className="button-group">
              <button
                className={`btn-toggle ${viewMode === "line" ? "active" : ""}`}
                onClick={() => setViewMode("line")}
              >
                Line
              </button>
              <button
                className={`btn-toggle ${viewMode === "area" ? "active" : ""}`}
                onClick={() => setViewMode("area")}
              >
                Area
              </button>
            </div>
          </div>
        </div>

        {chartData.length > 0 && (
          <ResponsiveContainer width="100%" height={450}>
            {viewMode === "line" ? (
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="date" stroke="#9ca3af" />
                <YAxis stroke="#9ca3af" />
                <Tooltip content={<CustomTooltip />} />
                <Line
                  type="monotone"
                  dataKey="AQI"
                  stroke="#1f77b4"
                  strokeWidth={3}
                  dot={{ r: 4 }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            ) : (
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="date" stroke="#9ca3af" />
                <YAxis stroke="#9ca3af" />
                <Tooltip content={<CustomTooltip />} />
                <Area
                  type="monotone"
                  dataKey="AQI"
                  stroke="#1f77b4"
                  strokeWidth={2}
                  fill="rgba(31,119,180,0.3)"
                />
              </AreaChart>
            )}
          </ResponsiveContainer>
        )}

        {loading && <p>Loading historical AQI...</p>}
      </div>
    </div>
  );
}

export default App;