import React, { useState } from 'react';
import { apiService } from '../services/api';
import '../styles/BatchAnalyzer.css';

function BatchAnalyzer() {
  const [texts, setTexts] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [processingTime, setProcessingTime] = useState(0);

  const handleAnalyze = async () => {
    const textList = texts.split('\n')
      .map(t => t.trim())
      .filter(t => t.length > 0);

    if (textList.length === 0) {
      setError('Please enter at least one text');
      return;
    }

    if (textList.length > 100) {
      setError('Maximum 100 texts allowed');
      return;
    }

    setLoading(true);
    setError('');
    setResults([]);

    try {
      const response = await apiService.analyzeBatch(textList);
      setResults(response.results);
      setProcessingTime(response.processing_time);
    } catch (err) {
      setError(err.message || 'Error analyzing batch');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setTexts('');
    setResults([]);
    setError('');
  };

  const exportResults = () => {
    const csv = results.map(r => 
      `"${r.text.replace(/"/g, '""')}","${r.sentiment}",${r.confidence}`
    ).join('\n');
    
    const header = '"Text","Sentiment","Confidence"\n';
    const element = document.createElement('a');
    element.setAttribute('href', 'data:text/csv;charset=utf-8,' + encodeURIComponent(header + csv));
    element.setAttribute('download', 'sentiment_analysis_results.csv');
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  return (
    <div className="batch-analyzer">
      <div className="input-section">
        <label htmlFor="batch-input">Enter texts (one per line):</label>
        <textarea
          id="batch-input"
          value={texts}
          onChange={(e) => setTexts(e.target.value)}
          placeholder="Enter multiple texts, one per line..."
          rows="10"
        />
        <div className="text-count">
          {texts.split('\n').filter(t => t.trim()).length} texts
        </div>

        <div className="button-group">
          <button 
            className="btn btn-primary" 
            onClick={handleAnalyze}
            disabled={loading || !texts.trim()}
          >
            {loading ? 'Analyzing...' : 'Analyze All'}
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

      {results.length > 0 && (
        <div className="results-section">
          <div className="results-header">
            <h2>Results ({results.length} texts analyzed)</h2>
            <span className="processing-time">Processing time: {processingTime.toFixed(2)}s</span>
            <button 
              className="btn btn-secondary"
              onClick={exportResults}
            >
              📥 Export CSV
            </button>
          </div>

          <div className="results-table-wrapper">
            <table className="results-table">
              <thead>
                <tr>
                  <th>Text</th>
                  <th>Sentiment</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {results.map((result, idx) => (
                  <tr key={idx}>
                    <td className="text-column">{result.text}</td>
                    <td className="sentiment-column">
                      <span className="emoji">{result.emoji}</span>
                      {result.sentiment}
                    </td>
                    <td className="confidence-column">
                      <div className="confidence-mini">
                        <div 
                          className="confidence-mini-fill"
                          style={{ width: `${result.confidence * 100}%` }}
                        ></div>
                      </div>
                      {(result.confidence * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

export default BatchAnalyzer;
