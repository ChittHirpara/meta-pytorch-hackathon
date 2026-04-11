'use strict';

const app = require('./app');
const env = require('./config/env');
const logger = require('./utils/logger');

const server = app.listen(env.PORT, '0.0.0.0', () => {
  logger.info(`NetPulse server running on http://0.0.0.0:${env.PORT}`);
  logger.info(`Environment: ${env.NODE_ENV}`);
  logger.info('Available endpoints: POST /reset | POST /step | GET /state | GET /tasks | GET /score | GET /health');
});

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received — shutting down gracefully');
  server.close(() => {
    logger.info('Server closed');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  logger.info('SIGINT received — shutting down');
  server.close(() => process.exit(0));
});

module.exports = server;
