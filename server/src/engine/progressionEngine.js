'use strict';

const stateManager = require('./stateManager');
const {
  HEALTH_DECAY_PER_STEP,
  ERROR_GROWTH_PER_STEP,
  QUEUE_GROWTH_PER_STEP,
  LATENCY_GROWTH_PER_STEP,
  THRESHOLDS,
} = require('./constants');
const { clamp, roundTo, averageSystemHealth } = require('../utils/scoreUtils');

/**
 * Advance the environment by one time step.
 * - Degrades unhealthy services passively
 * - Updates customer_impact and system_health
 * - Propagates cascade effects to dependents
 * - Checks episode termination
 *
 * Called AFTER the action has been applied.
 */
function advanceStep() {
  const state = stateManager.getState();

  // Passive degradation per service
  const updatedServices = state.services.map((svc) => {
    // Only degrade services that are below healthy threshold
    if (svc.health >= THRESHOLDS.HEALTH_GOOD) {
      // Healthy services slowly recover toward full health (capped at 1.0)
      return {
        ...svc,
        health: roundTo(clamp(svc.health + 0.01, 0, 1), 4),
        latency: Math.max(20, Math.round(svc.latency * 0.97)),
        error_rate: roundTo(clamp(svc.error_rate * 0.95, 0, 1), 4),
        queue_depth: Math.max(0, Math.round(svc.queue_depth * 0.9)),
      };
    }

    // Unhealthy services degrade further
    const newHealth = roundTo(clamp(svc.health - HEALTH_DECAY_PER_STEP, 0, 1), 4);
    const newError = roundTo(clamp(svc.error_rate + ERROR_GROWTH_PER_STEP, 0, 1), 4);
    const newQueue = Math.round(svc.queue_depth + QUEUE_GROWTH_PER_STEP);
    const newLatency = Math.round(svc.latency + LATENCY_GROWTH_PER_STEP);

    return {
      ...svc,
      health: newHealth,
      error_rate: newError,
      queue_depth: newQueue,
      latency: newLatency,
    };
  });

  // Cascade: services downstream of a critical dependency also suffer
  const degradedNames = updatedServices
    .filter((s) => s.health < THRESHOLDS.HEALTH_CRITICAL)
    .map((s) => s.name);

  const cascadedServices = updatedServices.map((svc) => {
    const hasSickDep = svc.dependencies.some((dep) => degradedNames.includes(dep));
    if (!hasSickDep) return svc;

    // Downstream service takes a mild hit from the sick dependency
    return {
      ...svc,
      health: roundTo(clamp(svc.health - 0.01, 0, 1), 4),
      error_rate: roundTo(clamp(svc.error_rate + 0.005, 0, 1), 4),
      latency: Math.round(svc.latency + 5),
    };
  });

  // Recompute system-wide health
  const systemHealth = averageSystemHealth(cascadedServices);

  // Customer impact: inverse of system health, scaled 0–100, capped
  // Impact rises quickly when health < 50%, more slowly above
  let newImpact = state.customer_impact;
  if (systemHealth < 0.5) {
    newImpact = clamp(newImpact + 3.0, 0, 100);
  } else if (systemHealth < THRESHOLDS.SYSTEM_HEALTH_STABILIZED) {
    newImpact = clamp(newImpact + 1.0, 0, 100);
  } else {
    // System recovering — impact slowly decreases
    newImpact = clamp(newImpact - 2.0, 0, 100);
  }

  stateManager.patchState({
    services: cascadedServices,
    system_health: roundTo(systemHealth, 4),
    customer_impact: roundTo(newImpact, 2),
    step_count: state.step_count + 1,
  });
}

/**
 * Check and return whether the episode should terminate.
 * @returns {{ done: boolean, success: boolean, reason: string }}
 */
function checkTermination() {
  const state = stateManager.getState();
  const { task_name, step_count, max_steps, remaining_budget, customer_impact,
          system_health, root_cause_resolved } = state;

  // Budget exhausted
  if (remaining_budget <= 0) {
    return { done: true, success: false, reason: 'budget_exhausted' };
  }

  // Max steps reached
  if (step_count >= max_steps) {
    return { done: true, success: false, reason: 'max_steps_reached' };
  }

  // Load success conditions from state (stored at reset)
  const sc = state.success_conditions;
  if (
    sc &&
    root_cause_resolved &&
    system_health >= sc.min_system_health &&
    customer_impact <= sc.max_customer_impact
  ) {
    return { done: true, success: true, reason: 'success' };
  }

  return { done: false, success: false, reason: 'in_progress' };
}

module.exports = { advanceStep, checkTermination };
