import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:5019/api'; // Backend API base URL

// Instance of axios with default settings
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  }
});

/**
 * Calls the /api/echarts endpoint.
 * @param {string} query - The user's current chat content.
 * @param {string} database - The database name (defaults to 'archive').
 * @returns {Promise<Object>} - The Echarts data from the API.
 */
export const getEchartsData = async (query, database = 'archive') => {
  try {
    const response = await apiClient.post('/echarts', { query, database });
    if (response.data && response.data.CODE === 20000) {
      return response.data.DATA;
    } else {
      console.error('Error from /api/echarts:', response.data);
      throw new Error(response.data.MSG || 'Failed to fetch Echarts data');
    }
  } catch (error) {
    console.error('Error calling /api/echarts:', error);
    throw error;
  }
};

/**
 * Calls the /api/query endpoint.
 * @param {string} query - The user's current chat content.
 * @param {string} database - The database name (defaults to 'archive').
 * @returns {Promise<Object>} - The query result from the API (text data and images).
 */
export const getQueryData = async (query, database = 'archive') => {
  try {
    const response = await apiClient.post('/query', { query, database });
    if (response.data && response.data.CODE === 20000) {
      return response.data.DATA;
    } else {
      console.error('Error from /api/query:', response.data);
      throw new Error(response.data.MSG || 'Failed to fetch query data');
    }
  } catch (error) {
    console.error('Error calling /api/query:', error);
    throw error;
  }
};

export default {
  getEchartsData,
  getQueryData
};
