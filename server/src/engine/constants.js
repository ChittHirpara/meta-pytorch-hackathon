'use strict';

// Episode limits
const MAX_STEPS = 50;
const MAX_BUDGET = 100;

// Per-step passive degradation (applied each step to unhealthy services)
const HEALTH_DECAY_PER_STEP = 0.01;
const ERROR_GROWTH_PER_STEP = 0.008;
const QUEUE_GROWTH_PER_STEP = 3;
const LATENCY_GROWTH_PER_STEP = 6;

// Action costs
const ACTION_COSTS = {
  inspect_service: 1,
  inspect_dependency: 1,
  restart_service: 10,
  scale_service: 15,
  rollback_deploy: 12,
  reroute_traffic: 8,
  throttle_queue: 5,
  silence_alert: 2,
  noop: 0,
};

// Action effects
const ACTION_EFFECTS = {
  restart_service: {
    health_boost: 0.35,
    latency_reduction: 0.4,
    error_rate_reduction: 0.5,
  },
  scale_service: {
    queue_reduction: 0.6,
    latency_reduction: 0.3,
    health_boost: 0.1,
  },
  rollback_deploy: {
    health_boost: 0.4,
    error_rate_reduction: 0.6,
    latency_reduction: 0.2,
  },
  reroute_traffic: {
    queue_reduction: 0.5,
    latency_reduction: 0.35,
  },
  throttle_queue: {
    queue_reduction: 0.7,
    latency_reduction: 0.2,
  },
};

// Severity levels
const SEVERITY = {
  CRITICAL: 'critical',
  HIGH: 'high',
  MEDIUM: 'medium',
  LOW: 'low',
  INFO: 'info',
};

// Service names
const SERVICES = [
  'auth_service',
  'api_gateway',
  'payment_worker',
  'notification_queue',
  'database_replica',
  'analytics_pipeline',
];

// Valid actions
const VALID_ACTIONS = [
  'inspect_service',
  'inspect_dependency',
  'restart_service',
  'scale_service',
  'rollback_deploy',
  'reroute_traffic',
  'throttle_queue',
  'silence_alert',
  'noop',
];

// Thresholds
const THRESHOLDS = {
  HEALTH_CRITICAL: 0.3,
  HEALTH_LOW: 0.5,
  HEALTH_GOOD: 0.8,
  ERROR_RATE_HIGH: 0.3,
  QUEUE_OVERLOAD: 100,
  CUSTOMER_IMPACT_MAX: 100,
  SYSTEM_HEALTH_STABILIZED: 0.75,
};

// Reward config
const REWARD = {
  INSPECT_USEFUL: 3,
  INSPECT_ALREADY_KNOWN: 0.5,
  HEALTH_IMPROVEMENT_MULTIPLIER: 20,
  CUSTOMER_IMPACT_REDUCTION_MULTIPLIER: 0.5,
  ROOT_CAUSE_RESOLVED: 25,
  CRITICAL_STABILIZED: 10,
  NOOP_PENALTY: -3,
  REPEATED_ACTION_PENALTY: -2,
  HARMFUL_ACTION_PENALTY: -8,
  BUDGET_WASTED_PENALTY: -1,
  SYMPTOM_MASK_PENALTY: -5,
};

// NEW: deterministic task variation profiles
const TASK_VARIATIONS = {
  easy_signal_noise: ['v1', 'v2', 'v3'],
  medium_hidden_dependency: ['v1', 'v2', 'v3'],
  hard_multi_incident: ['v1', 'v2', 'v3'],
};

module.exports = {
  MAX_STEPS,
  MAX_BUDGET,
  HEALTH_DECAY_PER_STEP,
  ERROR_GROWTH_PER_STEP,
  QUEUE_GROWTH_PER_STEP,
  LATENCY_GROWTH_PER_STEP,
  ACTION_COSTS,
  ACTION_EFFECTS,
  SEVERITY,
  SERVICES,
  VALID_ACTIONS,
  THRESHOLDS,
  REWARD,
  TASK_VARIATIONS,
};