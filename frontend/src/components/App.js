import React, { useState, useEffect } from 'react';
import Header from './Header';
import TextAnalyzer from './TextAnalyzer';
import BatchAnalyzer from './BatchAnalyzer';
import FileAnalyzer from './FileAnalyzer';
import Dashboard from './Dashboard';
import '../styles/App.css';

function App() {
  const [activeTab, setActiveTab] = useState('single');
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    // Check API health on mount
    checkAPIHealth();
  }, []);

  const checkAPIHealth = async () => {
    try {
      const response = await fetch(`https://siva0801-sentimentanalysis-api.hf.space/health`);
      if (response.ok) {
        setApiStatus('healthy');
      } else {
        setApiStatus('error');
      }
    } catch (error) {
      setApiStatus('error');
    }
  };

  return (
    <div className="app">
      <Header apiStatus={apiStatus} />
      
      <div className="container">
        {apiStatus === 'error' && (
          <div className="alert alert-error">
            ⚠️ Cannot connect to backend API. Make sure the backend server is running
          </div>
        )}

        <div className="tabs">
          <button 
            className={`tab-button ${activeTab === 'single' ? 'active' : ''}`}
            onClick={() => setActiveTab('single')}
          >
            📝 Single Text
          </button>
          <button 
            className={`tab-button ${activeTab === 'batch' ? 'active' : ''}`}
            onClick={() => setActiveTab('batch')}
          >
            📋 Batch Analysis
          </button>
          <button 
            className={`tab-button ${activeTab === 'file' ? 'active' : ''}`}
            onClick={() => setActiveTab('file')}
          >
            📁 Upload CSV
          </button>
          <button 
            className={`tab-button ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            📊 Dashboard
          </button>
        </div>

        <div className="tab-content">
          {activeTab === 'single' && <TextAnalyzer />}
          {activeTab === 'batch' && <BatchAnalyzer />}
          {activeTab === 'file' && <FileAnalyzer />}
          {activeTab === 'dashboard' && <Dashboard />}
        </div>
      </div>
    </div>
  );
}

export default App;
