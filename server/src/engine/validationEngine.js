'use strict';

const { ActionSchema } = require('../models/actionSchema');

/**
 * Validate an inbound action payload against the Zod ActionSchema.
 * Throws a structured error with status 400 on failure.
 * @param {object} payload - raw request body
 * @returns {{ action: string, target?: string }} parsed and validated action
 */
function validateAction(payload) {
  const result = ActionSchema.safeParse(payload);
  if (!result.success) {
    const messages = result.error.errors.map((e) => e.message).join('; ');
    const err = new Error(`Invalid action: ${messages}`);
    err.status = 400;
    err.details = result.error.errors;
    throw err;
  }
  return result.data;
}

module.exports = { validateAction };
