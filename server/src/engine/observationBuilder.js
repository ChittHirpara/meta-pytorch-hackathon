'use strict';

const { roundTo, averageSystemHealth } = require('../utils/scoreUtils');

/**
 * Build the observation object that is returned to the agent.
 * This is a curated view of state — internal-only fields (root_cause, is_real, etc.)
 * are NOT exposed here.
 *
 * @param {object} state - full internal state from stateManager
 * @returns {object} observation
 */
function buildObservation(state) {
  const services = state.services.map((svc) => ({
    name: svc.name,
    health: roundTo(svc.health, 4),
    latency: Math.round(svc.latency),
    error_rate: roundTo(svc.error_rate, 4),
    queue_depth: Math.round(svc.queue_depth),
    dependencies: svc.dependencies,
    status: deriveStatus(svc.health),
  }));

  const alerts = state.active_alerts
    .filter((a) => !a.silenced)
    .map((a) => ({
      id: a.id,
      service: a.service,
      severity: a.severity,
      message: a.message,
    }));

  return {
    task_name: state.task_name,
    task_variant: state.task_variant || 'v1',
    task_description: state.task_description || '',
    step_count: state.step_count,
    customer_impact: roundTo(state.customer_impact, 2),
    system_health: roundTo(averageSystemHealth(state.services), 4),
    remaining_budget: roundTo(state.remaining_budget, 2),
    recent_actions: state.recent_actions.slice(),
    services,
    alerts,
  };
}

/**
 * Derive a human-readable health status string.
 * @param {number} health 0–1
 * @returns {string}
 */
function deriveStatus(health) {
  if (health >= 0.85) return 'healthy';
  if (health >= 0.65) return 'degraded';
  if (health >= 0.40) return 'critical';
  return 'down';
}

module.exports = { buildObservation };