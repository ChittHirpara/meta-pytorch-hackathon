'use strict';

const path = require('path');
const fs = require('fs');
const { TaskSchema } = require('../models/taskSchema');
const logger = require('../utils/logger');
const { TASK_VARIATIONS } = require('./constants');

const TASKS_DIR = path.resolve(__dirname, '../data/tasks');

const TASK_FILES = {
  easy_signal_noise: 'easy_signal_noise.json',
  medium_hidden_dependency: 'medium_hidden_dependency.json',
  hard_multi_incident: 'hard_multi_incident.json',
};

const _cache = {};
const _rotation = {
  easy_signal_noise: 0,
  medium_hidden_dependency: 0,
  hard_multi_incident: 0,
};

function deepClone(obj) {
  return JSON.parse(JSON.stringify(obj));
}

/**
 * Apply deterministic scenario variation to a task.
 */
function applyVariation(taskName, task, variantName) {
  const t = deepClone(task);

  if (taskName === 'easy_signal_noise') {
    if (variantName === 'v2') {
      const api = t.services.find(s => s.name === 'api_gateway');
      if (api) {
        api.health = 0.28;
        api.latency = 1080;
        api.error_rate = 0.52;
        api.queue_depth = 162;
      }
      t.initial_budget = 55;
      t.alerts.push({
        id: 'alert-004',
        service: 'payment_worker',
        severity: 'low',
        message: 'Payment worker slight jitter observed. Possibly transient.',
        is_real: false
      });
    }

    if (variantName === 'v3') {
      const api = t.services.find(s => s.name === 'api_gateway');
      const notif = t.services.find(s => s.name === 'notification_queue');

      if (api) {
        api.health = 0.36;
        api.latency = 860;
        api.error_rate = 0.39;
        api.queue_depth = 118;
      }
      if (notif) {
        notif.latency = 95;
        notif.queue_depth = 38;
      }

      t.initial_budget = 65;
      t.alerts[1].severity = 'medium';
      t.alerts[1].message = 'Notification queue depth mildly elevated. Monitor before action.';
    }
  }

  if (taskName === 'medium_hidden_dependency') {
    if (variantName === 'v2') {
      const db = t.services.find(s => s.name === 'database_replica');
      const pay = t.services.find(s => s.name === 'payment_worker');

      if (db) {
        db.health = 0.24;
        db.latency = 2800;
        db.error_rate = 0.68;
        db.queue_depth = 360;
      }
      if (pay) {
        pay.queue_depth = 240;
        pay.latency = 1950;
      }

      t.initial_budget = 70;
    }

    if (variantName === 'v3') {
      const db = t.services.find(s => s.name === 'database_replica');
      const analytics = t.services.find(s => s.name === 'analytics_pipeline');

      if (db) {
        db.health = 0.34;
        db.latency = 2100;
        db.error_rate = 0.51;
        db.queue_depth = 270;
      }
      if (analytics) {
        analytics.health = 0.64;
        analytics.latency = 420;
        analytics.error_rate = 0.12;
      }

      t.alerts.push({
        id: 'alert-105',
        service: 'analytics_pipeline',
        severity: 'high',
        message: 'Analytics reporting SLA breach. Looks urgent but is downstream only.',
        is_real: false
      });
    }
  }

  if (taskName === 'hard_multi_incident') {
    if (variantName === 'v2') {
      const gateway = t.services.find(s => s.name === 'api_gateway');
      const payment = t.services.find(s => s.name === 'payment_worker');
      const notif = t.services.find(s => s.name === 'notification_queue');

      if (gateway) {
        gateway.health = 0.22;
        gateway.latency = 1650;
        gateway.error_rate = 0.67;
        gateway.queue_depth = 320;
      }
      if (payment) {
        payment.health = 0.34;
        payment.queue_depth = 225;
      }
      if (notif) {
        notif.health = 0.30;
        notif.queue_depth = 180;
      }

      t.initial_budget = 75;
    }

    if (variantName === 'v3') {
      const auth = t.services.find(s => s.name === 'auth_service');
      const gateway = t.services.find(s => s.name === 'api_gateway');

      if (auth) {
        auth.health = 0.48;
        auth.latency = 460;
        auth.error_rate = 0.24;
      }
      if (gateway) {
        gateway.health = 0.27;
        gateway.latency = 1420;
        gateway.error_rate = 0.58;
        gateway.queue_depth = 250;
      }

      t.alerts.push({
        id: 'alert-206',
        service: 'database_replica',
        severity: 'medium',
        message: 'Replica lag increased. Potentially distracting but not primary.',
        is_real: false
      });
    }
  }

  t.variant = variantName;
  return t;
}

/**
 * Load and validate a task by name.
 */
function loadTask(taskName) {
  if (!TASK_FILES[taskName]) {
    throw Object.assign(
      new Error(`Unknown task: '${taskName}'. Valid tasks: ${Object.keys(TASK_FILES).join(', ')}`),
      { status: 400 }
    );
  }

  if (!_cache[taskName]) {
    const filePath = path.join(TASKS_DIR, TASK_FILES[taskName]);

    if (!fs.existsSync(filePath)) {
      throw Object.assign(new Error(`Task file not found: ${filePath}`), { status: 500 });
    }

    let raw;
    try {
      raw = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    } catch (e) {
      throw Object.assign(new Error(`Failed to parse task file: ${e.message}`), { status: 500 });
    }

    const result = TaskSchema.safeParse(raw);
    if (!result.success) {
      logger.error('Task validation failed', result.error.format());
      throw Object.assign(new Error(`Task schema invalid: ${result.error.message}`), { status: 500 });
    }

    _cache[taskName] = result.data;
    logger.info(`Task loaded and cached: ${taskName}`);
  }

  const baseTask = deepClone(_cache[taskName]);

  // deterministic round-robin variation
  const variants = TASK_VARIATIONS[taskName] || ['v1'];
  const idx = _rotation[taskName] % variants.length;
  const chosenVariant = variants[idx];
  _rotation[taskName] += 1;

  const variedTask = applyVariation(taskName, baseTask, chosenVariant);
  logger.info(`Task variation selected: ${taskName} -> ${chosenVariant}`);

  return variedTask;
}

/**
 * Return available task names.
 */
function listTasks() {
  return Object.keys(TASK_FILES);
}

module.exports = { loadTask, listTasks };