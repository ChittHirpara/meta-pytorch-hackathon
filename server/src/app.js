'use strict';

const express = require('express');
const cors = require('cors');
const { requestLogger } = require('./middleware/requestLogger');
const { errorHandler } = require('./middleware/errorHandler');
const envRoutes = require('./routes/envRoutes');

const app = express();

// ── Middleware ──────────────────────────────────────────────────────────────
app.use(cors());
app.use(express.json());
app.use(requestLogger);

// ── Routes ──────────────────────────────────────────────────────────────────
app.use('/', envRoutes);

// 404 catch-all
app.use((req, res) => {
  res.status(404).json({
    ok: false,
    error: `Route not found: ${req.method} ${req.originalUrl}`,
    available_routes: [
      'POST /reset',
      'POST /step',
      'GET /state',
      'GET /tasks',
      'GET /score',
      'GET /health',
    ],
  });
});

// ── Error handler (must be last) ────────────────────────────────────────────
app.use(errorHandler);

module.exports = app;
