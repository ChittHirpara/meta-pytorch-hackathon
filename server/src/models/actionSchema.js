'use strict';

const { z } = require('zod');
const { VALID_ACTIONS, SERVICES } = require('../engine/constants');

/**
 * Zod schema for an inbound action from the agent.
 * Handles:
 * - service-targeted actions
 * - alert-targeted actions (silence_alert)
 * - no-target actions (noop)
 */
const ActionSchema = z
  .object({
    action: z.enum(VALID_ACTIONS, {
      errorMap: () => ({
        message: `action must be one of: ${VALID_ACTIONS.join(', ')}`,
      }),
    }),
    target: z.string().optional(),
  })
  .superRefine((data, ctx) => {
    const { action, target } = data;

    const serviceTargetActions = [
      'inspect_service',
      'inspect_dependency',
      'restart_service',
      'scale_service',
      'rollback_deploy',
      'reroute_traffic',
      'throttle_queue',
    ];

    const alertTargetActions = ['silence_alert'];
    const noTargetActions = ['noop'];

    // Service-targeted actions
    if (serviceTargetActions.includes(action)) {
      if (!target) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ['target'],
          message: `target is required for action '${action}'`,
        });
        return;
      }

      if (!SERVICES.includes(target)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ['target'],
          message: `target '${target}' is not a valid service. Valid services: ${SERVICES.join(', ')}`,
        });
      }
    }

    // Alert-targeted actions
    if (alertTargetActions.includes(action)) {
      if (!target) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ['target'],
          message: `target (alert id) is required for action '${action}'`,
        });
      }
    }

    // No-target actions
    if (noTargetActions.includes(action)) {
      if (target !== undefined && target !== '') {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ['target'],
          message: `target should not be provided for action '${action}'`,
        });
      }
    }
  });

module.exports = { ActionSchema };