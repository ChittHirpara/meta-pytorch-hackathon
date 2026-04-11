'use strict';

const stateManager = require('./stateManager');
const { ACTION_COSTS, ACTION_EFFECTS, THRESHOLDS, REWARD } = require('./constants');
const { clamp, roundTo } = require('../utils/scoreUtils');
const logger = require('../utils/logger');

/**
 * @param {{ action: string, target?: string }} validatedAction
 * @returns {{ applied: boolean, effects: object, warnings: string[] }}
 */
function applyAction(validatedAction) {
  const state = stateManager.getState();
  const { action, target } = validatedAction;
  const effects = {};
  const warnings = [];

  const cost = ACTION_COSTS[action] || 0;
  if (cost > 0 && state.remaining_budget < cost) {
    warnings.push(`Insufficient budget (need ${cost}, have ${roundTo(state.remaining_budget, 2)}). Action still applied at degraded effect.`);
  }
  const actualCost = Math.min(cost, state.remaining_budget);
  stateManager.patchState({ remaining_budget: roundTo(state.remaining_budget - actualCost, 4) });
  effects.budget_spent = actualCost;

  switch (action) {
    case 'inspect_service':
      effects.inspect = _handleInspect(target, false);
      break;
    case 'inspect_dependency':
      effects.inspect = _handleInspect(target, true);
      break;
    case 'restart_service':
      effects.restart = _handleRestart(target, state);
      break;
    case 'scale_service':
      effects.scale = _handleScale(target);
      break;
    case 'rollback_deploy':
      effects.rollback = _handleRollback(target, state);
      break;
    case 'reroute_traffic':
      effects.reroute = _handleReroute(target);
      break;
    case 'throttle_queue':
      effects.throttle = _handleThrottle(target);
      break;
    case 'silence_alert':
      effects.silence = _handleSilence(target, state);
      break;
    case 'noop':
      effects.noop = true;
      break;
    default:
      break;
  }

  _checkRootCauseResolved();

  return { applied: true, effects, warnings };
}


function _handleInspect(target, isDependency) {
  if (!target) return { skipped: true, reason: 'no target specified' };
  const svc = stateManager.getState().services.find((s) => s.name === target);
  if (!svc) return { skipped: true, reason: `service '${target}' not found` };

  const field = isDependency ? 'dependency_inspected' : 'inspected';
  const alreadyDone = svc[field];
  stateManager.updateService(target, { [field]: true });

  return { target, already_known: alreadyDone, dependency_check: isDependency };
}

function _handleRestart(target, priorState) {
  if (!target) return { skipped: true, reason: 'no target specified' };
  const fx = ACTION_EFFECTS.restart_service;

  const svc = priorState.services.find((s) => s.name === target);
  if (!svc) return { skipped: true, reason: `service '${target}' not found` };

  const newHealth = clamp(svc.health + fx.health_boost, 0, 1);
  const newLatency = clamp(svc.latency * (1 - fx.latency_reduction), 20, 5000);
  const newErrorRate = clamp(svc.error_rate * (1 - fx.error_rate_reduction), 0, 1);

  stateManager.updateService(target, {
    health: roundTo(newHealth, 4),
    latency: Math.round(newLatency),
    error_rate: roundTo(newErrorRate, 4),
  });

  const state = stateManager.getState();
  if (
    state.task_name === 'hard_multi_incident' &&
    target === 'auth_service'
  ) {
    logger.warn('Cascade trap triggered: restarting auth_service during gateway failure');
    const pw = state.services.find((s) => s.name === 'payment_worker');
    if (pw) {
      stateManager.updateService('payment_worker', {
        error_rate: roundTo(clamp(pw.error_rate * 1.6, 0, 1), 4),
        queue_depth: Math.round(pw.queue_depth * 1.8),
        health: roundTo(clamp(pw.health - 0.18, 0, 1), 4),
      });
    }
    stateManager.patchState({ cascade_triggered: true });
    return { target, cascade_triggered: true };
  }

  return { target, health_delta: roundTo(newHealth - svc.health, 4) };
}

function _handleScale(target) {
  if (!target) return { skipped: true, reason: 'no target specified' };
  const fx = ACTION_EFFECTS.scale_service;
  const state = stateManager.getState();
  const svc = state.services.find((s) => s.name === target);
  if (!svc) return { skipped: true, reason: `service '${target}' not found` };

  stateManager.updateService(target, {
    queue_depth: Math.round(svc.queue_depth * (1 - fx.queue_reduction)),
    latency: Math.round(svc.latency * (1 - fx.latency_reduction)),
    health: roundTo(clamp(svc.health + fx.health_boost, 0, 1), 4),
  });

  return { target, queue_reduced: true };
}

function _handleRollback(target, priorState) {
  if (!target) return { skipped: true, reason: 'no target specified' };
  const fx = ACTION_EFFECTS.rollback_deploy;
  const svc = priorState.services.find((s) => s.name === target);
  if (!svc) return { skipped: true, reason: `service '${target}' not found` };

  const newHealth = clamp(svc.health + fx.health_boost, 0, 1);
  const newErrorRate = clamp(svc.error_rate * (1 - fx.error_rate_reduction), 0, 1);
  const newLatency = clamp(svc.latency * (1 - fx.latency_reduction), 20, 5000);

  stateManager.updateService(target, {
    health: roundTo(newHealth, 4),
    error_rate: roundTo(newErrorRate, 4),
    latency: Math.round(newLatency),
  });

  return { target, rolled_back: true, health_delta: roundTo(newHealth - svc.health, 4) };
}

function _handleReroute(target) {
  if (!target) return { skipped: true, reason: 'no target specified' };
  const fx = ACTION_EFFECTS.reroute_traffic;
  const state = stateManager.getState();
  const svc = state.services.find((s) => s.name === target);
  if (!svc) return { skipped: true, reason: `service '${target}' not found` };

  stateManager.updateService(target, {
    queue_depth: Math.round(svc.queue_depth * (1 - fx.queue_reduction)),
    latency: Math.round(svc.latency * (1 - fx.latency_reduction)),
  });

  return { target, rerouted: true };
}

function _handleThrottle(target) {
  if (!target) return { skipped: true, reason: 'no target specified' };
  const fx = ACTION_EFFECTS.throttle_queue;
  const state = stateManager.getState();
  const svc = state.services.find((s) => s.name === target);
  if (!svc) return { skipped: true, reason: `service '${target}' not found` };

  stateManager.updateService(target, {
    queue_depth: Math.round(svc.queue_depth * (1 - fx.queue_reduction)),
    latency: Math.round(svc.latency * (1 - fx.latency_reduction)),
  });

  return { target, throttled: true };
}

function _handleSilence(target, priorState) {
  // target here is alertId
  const alertId = target;
  if (!alertId) return { skipped: true, reason: 'alert id required as target' };

  const alert = priorState.active_alerts.find((a) => a.id === alertId);
  if (!alert) return { skipped: true, reason: `alert '${alertId}' not found` };

  stateManager.silenceAlert(alertId);

  return { alert_id: alertId, was_real: alert.is_real, silenced: true };
}

function _checkRootCauseResolved() {
  const state = stateManager.getState();
  if (state.root_cause_resolved) return; // already resolved

  const rootSvc = state.services.find((s) => s.name === state.hidden_root_cause);
  if (rootSvc && rootSvc.health >= THRESHOLDS.HEALTH_LOW) {
    stateManager.patchState({ root_cause_resolved: true });
    logger.info(`Root cause resolved: ${state.hidden_root_cause}`);
  }
}

module.exports = { applyAction };
