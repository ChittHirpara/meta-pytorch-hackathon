'use strict';

const { deepClone } = require('../utils/deepClone');

/**
 * In-memory environment state store.
 * Single source of truth for the running episode.
 */
let _state = null;

/**
 * Initialise (or replace) the current state.
 * @param {object} initialState
 */
function setState(initialState) {
  _state = deepClone(initialState);
}

/**
 * Return a deep-cloned snapshot of the current state.
 * External callers never get a live reference.
 * @returns {object}
 */
function getState() {
  if (!_state) return null;
  return deepClone(_state);
}

/**
 * Apply a partial patch to the current state.
 * Merges top-level keys only — for nested mutations call
 * updateService / updateAlert helpers.
 * @param {object} patch
 */
function patchState(patch) {
  if (!_state) throw new Error('State not initialised. Call reset() first.');
  Object.assign(_state, patch);
}

/**
 * Update a single service within the state by name.
 * @param {string} serviceName
 * @param {object} patch
 */
function updateService(serviceName, patch) {
  if (!_state) throw new Error('State not initialised.');
  const svc = _state.services.find((s) => s.name === serviceName);
  if (!svc) throw new Error(`Service '${serviceName}' not found in state.`);
  Object.assign(svc, patch);
}

/**
 * Mark an alert as silenced by its ID.
 * @param {string} alertId
 */
function silenceAlert(alertId) {
  if (!_state) throw new Error('State not initialised.');
  const alert = _state.active_alerts.find((a) => a.id === alertId);
  if (alert) alert.silenced = true;
}

/**
 * Append an action record to history.
 * @param {{ step: number, action: string, target?: string, reward: number }} record
 */
function appendActionHistory(record) {
  if (!_state) throw new Error('State not initialised.');
  _state.action_history.push(record);
  // Keep recent_actions as last 5 action+target strings for observation
  const label = record.target
    ? `${record.action}:${record.target}`
    : record.action;
  _state.recent_actions.unshift(label);
  if (_state.recent_actions.length > 5) {
    _state.recent_actions = _state.recent_actions.slice(0, 5);
  }
}

/**
 * Check whether state is initialised.
 * @returns {boolean}
 */
function isReady() {
  return _state !== null;
}

/**
 * Completely wipe the state (used by /reset).
 */
function clearState() {
  _state = null;
}

module.exports = {
  setState,
  getState,
  patchState,
  updateService,
  silenceAlert,
  appendActionHistory,
  isReady,
  clearState,
};
