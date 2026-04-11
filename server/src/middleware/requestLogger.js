'use strict';

const logger = require('../utils/logger');

/**
 * Lightweight request logger — logs method, path, status, duration.
 */
function requestLogger(req, res, next) {
  const start = Date.now();
  res.on('finish', () => {
    const duration = Date.now() - start;
    logger.info(`${req.method} ${req.originalUrl} → ${res.statusCode} (${duration}ms)`);
  });
  next();
}

module.exports = { requestLogger };