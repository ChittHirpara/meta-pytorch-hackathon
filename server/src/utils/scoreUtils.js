'use strict';

/**
 * Clamp a value between min and max (inclusive).
 * @param {number} val
 * @param {number} min
 * @param {number} max
 * @returns {number}
 */
function clamp(val, min, max) {
  return Math.min(Math.max(val, min), max);
}

/**
 * Round a float to N decimal places for deterministic output.
 * @param {number} val
 * @param {number} decimals
 * @returns {number}
 */
function roundTo(val, decimals = 4) {
  const factor = Math.pow(10, decimals);
  return Math.round(val * factor) / factor;
}

/**
 * Compute average system health from services array.
 * @param {Array} services
 * @returns {number} 0–1
 */
function averageSystemHealth(services) {
  if (!services || services.length === 0) return 0;
  const sum = services.reduce((acc, s) => acc + s.health, 0);
  return roundTo(sum / services.length, 4);
}

/**
 * Normalise a raw reward to a safe range.
 * @param {number} raw
 * @returns {number}
 */
function normaliseReward(raw) {
  return roundTo(clamp(raw, -100, 100), 4);
}

module.exports = { clamp, roundTo, averageSystemHealth, normaliseReward };