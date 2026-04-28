import React, { useState, useEffect } from 'react';
import { apiService } from '../services/api';
import '../styles/Dashboard.css';

function Dashboard() {
  const [stats, setStats] = useState(null);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    setLoading(true);
    setError('');

    try {
      const [statsData, statusData] = await Promise.all([
        apiService.getModelStats(),
        apiService.getStatus()
      ]);
      
      setStats(statsData);
      setStatus(statusData);
    } catch (err) {
      setError('Error loading dashboard data');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="dashboard"><p>Loading...</p></div>;
  }

  return (
    <div className="dashboard">
      {error && <div className="alert alert-error">{error}</div>}

      <div className="dashboard-grid">
        <div className="card">
          <h3>📊 Model Information</h3>
          <div className="card-content">
            {stats && (
              <>
                <div className="info-row">
                  <span>Status:</span>
                  <strong>{stats.status}</strong>
                </div>
                <div className="info-row">
                  <span>Model Type:</span>
                  <strong>{stats.model_type}</strong>
                </div>
                <div className="info-row">
                  <span>Vectorizer:</span>
                  <strong>{stats.vectorizer}</strong>
                </div>
                <div className="info-row">
                  <span>Max Features:</span>
                  <strong>{stats.max_features}</strong>
                </div>
                <div className="info-row">
                  <span>Trained:</span>
                  <strong>{stats.is_trained ? '✅ Yes' : '❌ No'}</strong>
                </div>
              </>
            )}
          </div>
        </div>

        <div className="card">
          <h3>🔧 API Status</h3>
          <div className="card-content">
            {status && (
              <>
                <div className="info-row">
                  <span>API Name:</span>
                  <strong>{status.api}</strong>
                </div>
                <div className="info-row">
                  <span>Version:</span>
                  <strong>{status.version}</strong>
                </div>
                <div className="info-row">
                  <span>Status:</span>
                  <strong className="status-running">{status.status}</strong>
                </div>
                <div className="info-row">
                  <span>Model Type:</span>
                  <strong>{status.model?.type}</strong>
                </div>
                <div className="info-row">
                  <span>Vectorizer:</span>
                  <strong>{status.model?.vectorizer}</strong>
                </div>
              </>
            )}
          </div>
        </div>

        <div className="card full-width">
          <h3>📈 Features</h3>
          <div className="card-content features-list">
            <div className="feature">
              <span>✅</span>
              <span>AI-Powered Sentiment Analysis</span>
            </div>
            <div className="feature">
              <span>✅</span>
              <span>Single Text Analysis</span>
            </div>
            <div className="feature">
              <span>✅</span>
              <span>Batch Processing (up to 100 texts)</span>
            </div>
            <div className="feature">
              <span>✅</span>
              <span>CSV File Upload</span>
            </div>
            <div className="feature">
              <span>✅</span>
              <span>Confidence Scores</span>
            </div>
            <div className="feature">
              <span>✅</span>
              <span>Export Results</span>
            </div>
          </div>
        </div>

        <div className="card full-width">
          <h3>🚀 Getting Started</h3>
          <div className="card-content">
            <ol>
              <li>Go to the <strong>Single Text</strong> tab to analyze individual reviews</li>
              <li>Use <strong>Batch Analysis</strong> for multiple texts</li>
              <li>Upload a CSV file to analyze many reviews at once</li>
              <li>Export your results in CSV format</li>
            </ol>
          </div>
        </div>
      </div>

      <div className="refresh-section">
        <button className="btn btn-secondary" onClick={fetchDashboardData}>
          🔄 Refresh Data
        </button>
      </div>
    </div>
  );
}

export default Dashboard;
