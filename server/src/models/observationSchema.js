'use strict';

const { z } = require('zod');

const ServiceObsSchema = z.object({
  name: z.string(),
  health: z.number().min(0).max(1),
  latency: z.number().min(0),
  error_rate: z.number().min(0).max(1),
  queue_depth: z.number().min(0),
  dependencies: z.array(z.string()),
  status: z.string(),
});

const AlertSchema = z.object({
  id: z.string(),
  service: z.string(),
  severity: z.enum(['critical', 'high', 'medium', 'low', 'info']),
  message: z.string(),
  silenced: z.boolean().optional(),
});

const ObservationSchema = z.object({
  task_name: z.string(),
  step_count: z.number().int().min(0),
  customer_impact: z.number().min(0).max(100),
  system_health: z.number().min(0).max(1),
  remaining_budget: z.number().min(0),
  recent_actions: z.array(z.string()),
  services: z.array(ServiceObsSchema),
  alerts: z.array(AlertSchema),
});

module.exports = { ObservationSchema, ServiceObsSchema, AlertSchema };