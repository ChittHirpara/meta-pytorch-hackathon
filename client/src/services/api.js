import axios from 'axios';

// Connect to local backend for OpenEnv
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5050';

const api = axios.create({
  baseURL: API_URL,
});

export const getHealth = async () => {
  const res = await api.get('/health');
  return res.data;
};

export const getTasks = async () => {
  const res = await api.get('/tasks');
  return res.data;
};

export const resetEnv = async (task_name) => {
  const res = await api.post('/reset', { task_name });
  return res.data;
};

export const stepEnv = async (actionPayload) => {
  const res = await api.post('/step', actionPayload);
  return res.data;
};

export const getState = async () => {
  const res = await api.get('/state');
  return res.data;
};

export const getScore = async () => {
  const res = await api.get('/score');
  return res.data;
};
