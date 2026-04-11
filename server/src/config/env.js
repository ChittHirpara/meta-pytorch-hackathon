'use strict';

require('dotenv').config();

const env = {
  PORT: parseInt(process.env.PORT || '7860', 10),
  NODE_ENV: process.env.NODE_ENV || 'development',
  LOG_LEVEL: process.env.LOG_LEVEL || 'info',
  MAX_STEPS: parseInt(process.env.MAX_STEPS || '50', 10),
};

module.exports = env;
