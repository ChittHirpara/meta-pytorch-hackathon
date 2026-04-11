'use strict';

const { z } = require('zod');

const RewardSchema = z.object({
  reward: z.number(),
  reward_breakdown: z.object({
    health_improvement: z.number(),
    customer_impact_reduction: z.number(),
    diagnostic_value: z.number(),
    root_cause_progress: z.number(),
    penalties: z.number(),
  }),
  cumulative_reward: z.number(),
});

module.exports = { RewardSchema };