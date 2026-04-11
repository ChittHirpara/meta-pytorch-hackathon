export const ACTIONS = [
  'noop',
  'inspect_service',
  'inspect_dependency',
  'restart_service',
  'throttle_queue',
  'rollback_deploy',
  'scale_service',
];

export const getHealthColor = (health) => {
  if (health >= 0.8) return 'bg-emerald-500 shadow-emerald-500/50';
  if (health >= 0.5) return 'bg-amber-400 shadow-amber-400/50';
  return 'bg-rose-500 shadow-rose-500/50';
};

export const getHealthText = (health) => {
  if (health >= 0.8) return 'text-emerald-400';
  if (health >= 0.5) return 'text-amber-400';
  return 'text-rose-500';
};
