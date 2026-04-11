'use strict';

/**
 * Build a standard success response envelope.
 * @param {object} data
 * @returns {object}
 */
function success(data) {
  return { ok: true, ...data };
}

/**
 * Build a standard error response envelope.
 * @param {string} message
 * @param {object} [details]
 * @returns {object}
 */
function error(message, details = {}) {
  return { ok: false, error: message, ...details };
}

module.exports = { success, error };