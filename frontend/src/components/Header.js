import React from 'react';

function Header({ apiStatus }) {
  return (
    <header className="header">
      <div className="header-content">
        <h1>📊 Sentiment Analyzer</h1>
        <p>Analyze sentiment in product reviews using AI</p>
        <div className={`status-badge ${apiStatus}`}>
          {apiStatus === 'healthy' && '🟢 API Connected'}
          {apiStatus === 'checking' && '🟡 Checking...'}
          {apiStatus === 'error' && '🔴 API Offline'}
        </div>
      </div>
    </header>
  );
}

export default Header;
