'use strict';

const stateManager = require('./stateManager');
const { clamp, roundTo, averageSystemHealth } = require('../utils/scoreUtils');
const scoringMeta = require('../data/metadata/scoring.json');

/**
 * Compute the final deterministic grade for a completed episode.
 * Returns a score in [0.0, 1.0].
 *
 * Grading dimensions (weights come from scoring.json):
 *   1. System stability       (30%) — final avg health vs. target
 *   2. Customer harm reduction (25%) — how much impact was reduced
 *   3. Root cause resolution  (25%) — was root cause actually fixed
 *   4. Action efficiency      (10%) — steps used vs. max allowed
 *   5. Budget utilisation     (5%)  — budget remaining (not overspent)
 *   6. Harmful action avoidance (5%) — cascade traps, silenced real alerts
 *
 * @returns {{ score: number, breakdown: object, passed: boolean }}
 */
function computeFinalScore() {
  const state = stateManager.getState();
  const w = scoringMeta.weights;
  const thresholds = scoringMeta.thresholds;

  const finalHealth = averageSystemHealth(state.services);
  let stabilityScore;
  if (finalHealth >= thresholds.full_score_system_health) {
    stabilityScore = 1.0;
  } else if (finalHealth >= thresholds.pass_system_health) {
    // Linear interpolation between pass and full
    stabilityScore = (finalHealth - thresholds.pass_system_health) /
      (thresholds.full_score_system_health - thresholds.pass_system_health);
  } else {
    // Below pass threshold — linear down to 0
    stabilityScore = finalHealth / thresholds.pass_system_health * 0.5;
  }
  stabilityScore = clamp(stabilityScore, 0, 1);

  const finalImpact = state.customer_impact;
  const sc = state.success_conditions;
  const initialImpact = state.initial_customer_impact || 70; // stored at reset
  let customerScore;
  if (finalImpact <= thresholds.full_score_customer_impact) {
    customerScore = 1.0;
  } else if (finalImpact <= thresholds.pass_customer_impact) {
    customerScore = 1.0 - (finalImpact - thresholds.full_score_customer_impact) /
      (thresholds.pass_customer_impact - thresholds.full_score_customer_impact) * 0.5;
  } else {
    // Impact still high — partial credit for any reduction
    const reduction = clamp(initialImpact - finalImpact, 0, initialImpact);
    customerScore = (reduction / initialImpact) * 0.4;
  }
  customerScore = clamp(customerScore, 0, 1);

  const rootCauseScore = state.root_cause_resolved ? 1.0 : 0.0;

  // Fewer steps used = higher efficiency
  const stepsUsed = state.step_count;
  const maxSteps = state.max_steps;
  const efficiencyScore = clamp(1.0 - (stepsUsed / maxSteps), 0, 1);

  // Reward for having budget left — but not penalise too much for spending
  const budgetRatio = state.remaining_budget / state.initial_budget;
  const budgetScore = clamp(budgetRatio * 1.5, 0, 1); // generously scaled

  let harmScore = 1.0;
  if (state.cascade_triggered) {
    harmScore += scoringMeta.penalties.cascade_triggered; // subtract 0.15
  }
  const silencedRealAlerts = state.active_alerts.filter(
    (a) => a.silenced && a.is_real
  ).length;
  harmScore -= silencedRealAlerts * Math.abs(scoringMeta.penalties.silent_real_alert);
  if (state.reason === 'budget_exhausted') {
    harmScore += scoringMeta.penalties.budget_exhausted;
  }
  if (state.reason === 'max_steps_reached') {
    harmScore += scoringMeta.penalties.max_steps_reached;
  }
  harmScore = clamp(harmScore, 0, 1);

  const raw =
    w.system_stability * stabilityScore +
    w.customer_harm_reduction * customerScore +
    w.root_cause_resolution * rootCauseScore +
    w.action_efficiency * efficiencyScore +
    w.budget_utilisation * budgetScore +
    w.harmful_action_avoidance * harmScore;

  const score = roundTo(clamp(raw, 0, 1), 4);

  return {
    score,
    breakdown: {
      system_stability: roundTo(stabilityScore, 4),
      customer_harm_reduction: roundTo(customerScore, 4),
      root_cause_resolution: rootCauseScore,
      action_efficiency: roundTo(efficiencyScore, 4),
      budget_utilisation: roundTo(budgetScore, 4),
      harmful_action_avoidance: roundTo(harmScore, 4),
    },
    passed: score >= 0.5,
    root_cause_resolved: state.root_cause_resolved,
    final_health: roundTo(finalHealth, 4),
    final_customer_impact: roundTo(finalImpact, 2),
  };
}

module.exports = { computeFinalScore };
