import React, { useState } from 'react';
import { apiService } from '../services/api';
import '../styles/TextAnalyzer.css';

function TextAnalyzer() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await apiService.analyzeSentiment(text);
      setResult(response);
    } catch (err) {
      setError(err.message || 'Error analyzing sentiment');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setText('');
    setResult(null);
    setError('');
  };

  return (
    <div className="text-analyzer">
      <div className="input-section">
        <label htmlFor="text-input">Enter your text:</label>
        <textarea
          id="text-input"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter product review or any text to analyze sentiment..."
          rows="6"
          maxLength="10000"
        />
        <div className="char-count">{text.length} / 10000</div>

        <div className="button-group">
          <button 
            className="btn btn-primary" 
            onClick={handleAnalyze}
            disabled={loading || !text.trim()}
          >
            {loading ? 'Analyzing...' : 'Analyze Sentiment'}
          </button>
          <button 
            className="btn btn-secondary" 
            onClick={handleClear}
          >
            Clear
          </button>
        </div>
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      {result && (
        <div className="result-section">
          <h2>Analysis Result</h2>
          <div className="result-card">
            <div className="sentiment-display">
              <div className="emoji">{result.emoji}</div>
              <div className="sentiment-label">{result.sentiment}</div>
            </div>
            <div className="confidence-section">
              <label>Confidence Score:</label>
              <div className="confidence-bar">
                <div 
                  className="confidence-fill"
                  style={{ width: `${result.confidence * 100}%` }}
                ></div>
              </div>
              <span className="confidence-value">
                {(result.confidence * 100).toFixed(2)}%
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default TextAnalyzer;
