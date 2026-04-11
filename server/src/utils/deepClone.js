'use strict';

/**
 * Deterministic deep clone via JSON serialization.
 * Sufficient for plain-object state — no circular refs in our state model.
 * @param {*} obj
 * @returns {*}
 */
function deepClone(obj) {
  if (obj === undefined) return undefined;
  return JSON.parse(JSON.stringify(obj));
}

module.exports = { deepClone };