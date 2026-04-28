import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiService = {
  /**
   * Analyze sentiment for a single text
   */
  analyzeSentiment: async (text) => {
    try {
      const response = await api.post('/analyze', { text });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Error analyzing sentiment');
    }
  },

  /**
   * Analyze sentiment for multiple texts
   */
  analyzeBatch: async (texts) => {
    try {
      const response = await api.post('/analyze/batch', { texts });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Error in batch analysis');
    }
  },

  /**
   * Analyze sentiment from CSV file
   */
  analyzeCSV: async (file) => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await api.post('/analyze/csv', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Error uploading CSV file');
    }
  },

  /**
   * Get model statistics
   */
  getModelStats: async () => {
    try {
      const response = await api.get('/model/stats');
      return response.data;
    } catch (error) {
      throw new Error('Error fetching model stats');
    }
  },

  /**
   * Get API status
   */
  getStatus: async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/v1/status');
      return response.data;
    } catch (error) {
      throw new Error('Error fetching status');
    }
  },

  /**
   * Preprocess text (for debugging)
   */
  preprocessText: async (text) => {
    try {
      const response = await api.get('/preprocess', {
        params: { text },
      });
      return response.data;
    } catch (error) {
      throw new Error('Error preprocessing text');
    }
  },

  /**
   * Train the model (admin only)
   */
  trainModel: async () => {
    try {
      const response = await api.post('/train');
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.detail || 'Error training model');
    }
  },
};

export default api;
