'use strict';

const logger = require('../utils/logger');

/**
 * Centralised error handler. Always returns clean JSON.
 */
function errorHandler(err, req, res, next) {
  const status = err.status || err.statusCode || 500;
  const message = err.message || 'Internal server error';

  logger.error(`[${req.method}] ${req.path} → ${status}: ${message}`);

  return res.status(status).json({
    ok: false,
    error: message,
    ...(process.env.NODE_ENV !== 'production' && err.details
      ? { details: err.details }
      : {}),
  });
}

module.exports = { errorHandler };