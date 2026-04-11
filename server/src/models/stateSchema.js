'use strict';

const { z } = require('zod');

const ServiceStateSchema = z.object({
  name: z.string(),
  health: z.number().min(0).max(1),
  latency: z.number().min(0),
  error_rate: z.number().min(0).max(1),
  queue_depth: z.number().min(0),
  dependencies: z.array(z.string()),
  inspected: z.boolean(),
  dependency_inspected: z.boolean(),
});

const AlertStateSchema = z.object({
  id: z.string(),
  service: z.string(),
  severity: z.enum(['critical', 'high', 'medium', 'low', 'info']),
  message: z.string(),
  is_real: z.boolean(),
  silenced: z.boolean(),
});

const StateSchema = z.object({
  task_name: z.string(),
  step_count: z.number().int().min(0),
  customer_impact: z.number().min(0).max(100),
  system_health: z.number().min(0).max(1),
  remaining_budget: z.number().min(0),
  active_alerts: z.array(AlertStateSchema),
  hidden_root_cause: z.string(),
  root_cause_resolved: z.boolean(),
  recent_actions: z.array(z.string()),
  action_history: z.array(z.object({
    step: z.number(),
    action: z.string(),
    target: z.string().optional(),
    reward: z.number(),
  })),
  cumulative_reward: z.number(),
  score: z.number().min(0).max(1),
  done: z.boolean(),
  success: z.boolean(),
  services: z.array(ServiceStateSchema),
  max_steps: z.number().int().positive(),
  initial_budget: z.number().positive(),
});

module.exports = { StateSchema, ServiceStateSchema, AlertStateSchema };