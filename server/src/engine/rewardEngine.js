'use strict';

const stateManager = require('./stateManager');
const { REWARD, THRESHOLDS } = require('./constants');
const { roundTo, clamp, averageSystemHealth } = require('../utils/scoreUtils');

/**
 * Compute the step reward for the action just applied.
 * This is non-trivial, non-binary, and deterministic.
 *
 * @param {object} prevState   - deep clone of state BEFORE the action
 * @param {{ action: string, target?: string }} action - validated action
 * @param {object} effects     - output from actionHandler.applyAction()
 * @returns {{ reward: number, breakdown: object }}
 */
function computeReward(prevState, action, effects) {
  const currState = stateManager.getState();

  let healthImprovement = 0;
  let customerImpactReduction = 0;
  let diagnosticValue = 0;
  let rootCauseProgress = 0;
  let penalties = 0;

  // ── 1. Health improvement reward ────────────────────────────────────────────
  const prevSysHealth = averageSystemHealth(prevState.services);
  const currSysHealth = averageSystemHealth(currState.services);
  const healthDelta = currSysHealth - prevSysHealth;

  if (healthDelta > 0) {
    healthImprovement = roundTo(healthDelta * REWARD.HEALTH_IMPROVEMENT_MULTIPLIER, 4);
  } else if (healthDelta < -0.01) {
    // Action made things worse
    penalties += roundTo(healthDelta * 10, 4); // negative contribution
  }

  // ── 2. Customer impact reduction ─────────────────────────────────────────────
  const impactDelta = prevState.customer_impact - currState.customer_impact;
  if (impactDelta > 0) {
    customerImpactReduction = roundTo(impactDelta * REWARD.CUSTOMER_IMPACT_REDUCTION_MULTIPLIER, 4);
  }

  // ── 3. Diagnostic value (inspect actions) ───────────────────────────────────
  if (action.action === 'inspect_service' || action.action === 'inspect_dependency') {
    const inspectResult = effects.effects?.inspect;
    if (inspectResult && !inspectResult.skipped) {
      diagnosticValue = inspectResult.already_known
        ? REWARD.INSPECT_ALREADY_KNOWN
        : REWARD.INSPECT_USEFUL;
    }
  }

  // ── 4. Root cause progress ──────────────────────────────────────────────────
  if (!prevState.root_cause_resolved && currState.root_cause_resolved) {
    rootCauseProgress = REWARD.ROOT_CAUSE_RESOLVED;
  }

  // Check if action targeted root cause specifically and boosted its health
  if (action.target === currState.hidden_root_cause) {
    const prevRootSvc = prevState.services.find((s) => s.name === currState.hidden_root_cause);
    const currRootSvc = currState.services.find((s) => s.name === currState.hidden_root_cause);
    if (prevRootSvc && currRootSvc) {
      const rootHealthDelta = currRootSvc.health - prevRootSvc.health;
      if (rootHealthDelta > 0.05) {
        rootCauseProgress += roundTo(rootHealthDelta * 15, 4);
      }
    }
  }

  // ── 5. Penalties ────────────────────────────────────────────────────────────

  // Noop penalty
  if (action.action === 'noop') {
    penalties += REWARD.NOOP_PENALTY;
  }

  // Repeated action penalty (same action+target in last 3 steps)
  const recentCount = prevState.action_history
    .slice(-3)
    .filter(
      (h) => h.action === action.action && h.target === (action.target || '')
    ).length;
  if (recentCount >= 2) {
    penalties += REWARD.REPEATED_ACTION_PENALTY;
  }

  // Cascade triggered penalty
  if (effects.effects?.restart?.cascade_triggered) {
    penalties += REWARD.HARMFUL_ACTION_PENALTY;
  }

  // Silenced a real alert penalty
  if (effects.effects?.silence?.was_real) {
    penalties += REWARD.SYMPTOM_MASK_PENALTY;
  }

  // Budget wasted on a no-op service
  if (effects.effects?.budget_spent > 0 && healthDelta <= 0 && action.action !== 'inspect_service' && action.action !== 'inspect_dependency') {
    penalties += REWARD.BUDGET_WASTED_PENALTY;
  }

  // ── 6. Total ────────────────────────────────────────────────────────────────
  const reward = roundTo(
    healthImprovement + customerImpactReduction + diagnosticValue + rootCauseProgress + penalties,
    4
  );

  return {
    reward,
    breakdown: {
      health_improvement: roundTo(healthImprovement, 4),
      customer_impact_reduction: roundTo(customerImpactReduction, 4),
      diagnostic_value: roundTo(diagnosticValue, 4),
      root_cause_progress: roundTo(rootCauseProgress, 4),
      penalties: roundTo(penalties, 4),
    },
  };
}

module.exports = { computeReward };
