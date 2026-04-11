'use strict';

const { z } = require('zod');

const TaskSchema = z.object({
  name: z.string(),
  difficulty: z.enum(['easy', 'medium', 'hard']),
  description: z.string(),
  max_steps: z.number().int().positive(),
  initial_budget: z.number().positive(),
  hidden_root_cause: z.string(),
  success_conditions: z.object({
    min_system_health: z.number().min(0).max(1),
    max_customer_impact: z.number().min(0).max(100),
    root_cause_resolved: z.boolean(),
  }),
  services: z.array(z.object({
    name: z.string(),
    health: z.number().min(0).max(1),
    latency: z.number().min(0),
    error_rate: z.number().min(0).max(1),
    queue_depth: z.number().min(0),
    dependencies: z.array(z.string()),
  })),
  alerts: z.array(z.object({
    id: z.string(),
    service: z.string(),
    severity: z.enum(['critical', 'high', 'medium', 'low', 'info']),
    message: z.string(),
    is_real: z.boolean(),
  })),
});

module.exports = { TaskSchema };