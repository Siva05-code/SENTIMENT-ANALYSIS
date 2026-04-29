import React, { useState } from 'react';
import { apiService } from '../services/api';
import '../styles/FileAnalyzer.css';

function FileAnalyzer() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [fileName, setFileName] = useState('');
  const [results, setResults] = useState(null);
  const [downloadUrl, setDownloadUrl] = useState('');

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (!selectedFile.name.endsWith('.csv')) {
        setError('Please select a CSV file');
        setFile(null);
        return;
      }
      setFile(selectedFile);
      setFileName(selectedFile.name);
      setError('');
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');
    setResults(null);

    try {
      const response = await apiService.analyzeCSV(file);
      setSuccess(`✅ Successfully processed ${response.rows_processed} rows`);
      setResults(response.preview);
      setDownloadUrl(response.download_url);
      setFile(null);
      setFileName('');
    } catch (err) {
      setError(err.message || 'Error uploading file');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setFile(null);
    setFileName('');
    setError('');
    setSuccess('');
    setResults(null);
    setDownloadUrl('');
  };

  return (
    <div className="file-analyzer">
      <div className="upload-section">
        <div className="upload-box">
          <div className="upload-icon">📁</div>
          <h2>Upload CSV File</h2>
          <p>Required columns: ProductId, UserId, Text</p>
          
          <div className="file-input-wrapper">
            <input
              type="file"
              id="file-input"
              accept=".csv"
              onChange={handleFileChange}
              disabled={loading}
            />
            <label htmlFor="file-input" className="file-label">
              Choose File or Drag & Drop
            </label>
          </div>

          {fileName && (
            <div className="file-info">
              📄 {fileName}
            </div>
          )}

          <div className="button-group">
            <button 
              className="btn btn-primary" 
              onClick={handleUpload}
              disabled={loading || !file}
            >
              {loading ? 'Processing...' : 'Analyze CSV'}
            </button>
            <button 
              className="btn btn-secondary" 
              onClick={handleClear}
            >
              Clear
            </button>
          </div>
        </div>
      </div>

      {error && <div className="alert alert-error">{error}</div>}
      {success && <div className="alert alert-success">{success}</div>}

      {downloadUrl && (
        <div className="results-section">
          <div className="results-header">
            <h3>📊 Analysis Results</h3>
            <a href={downloadUrl} download className="btn btn-download">
              ⬇️ Download CSV
            </a>
          </div>
          
          {results && results.length > 0 && (
            <div className="results-preview">
              <h4>Preview (First 10 rows)</h4>
              <div className="table-wrapper">
                <table className="results-table">
                  <thead>
                    <tr>
                      {Object.keys(results[0]).map((key) => (
                        <th key={key}>{key}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((row, idx) => (
                      <tr key={idx}>
                        {Object.values(row).map((value, i) => (
                          <td key={i}>{String(value).substring(0, 50)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      <div className="info-section">
        <h3>CSV Format Requirements:</h3>
        <ul>
          <li><strong>ProductId:</strong> Unique identifier for the product</li>
          <li><strong>UserId:</strong> Unique identifier for the user</li>
          <li><strong>Text:</strong> Product review text</li>
        </ul>
        <h3>Example:</h3>
        <pre>
ProductId,UserId,Text
B001,U123,"Great product, highly recommend"
B002,U124,"Not satisfied with quality"
        </pre>
      </div>
    </div>
  );
}

export default FileAnalyzer;
