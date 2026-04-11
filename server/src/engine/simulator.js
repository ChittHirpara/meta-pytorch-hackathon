'use strict';

const stateManager = require('./stateManager');
const taskLoader = require('./taskLoader');
const observationBuilder = require('./observationBuilder');
const actionHandler = require('./actionHandler');
const progressionEngine = require('./progressionEngine');
const rewardEngine = require('./rewardEngine');
const grader = require('./grader');
const { deepClone } = require('../utils/deepClone');
const { roundTo, averageSystemHealth } = require('../utils/scoreUtils');
const logger = require('../utils/logger');

/**
 * Reset the environment with a given task.
 * Adds deterministic task variation metadata.
 * @param {string} taskName
 * @returns {{ observation: object }}
 */
function reset(taskName) {
  const task = taskLoader.loadTask(taskName);

  // Compute initial customer impact from initial service health
  const initialHealth = averageSystemHealth(task.services);
  const initialCustomerImpact = roundTo(Math.max(0, (1 - initialHealth) * 100), 2);

  // Build full internal state
  const initialState = {
    task_name: task.name,
    task_variant: task.variant || 'v1',
    step_count: 0,
    customer_impact: initialCustomerImpact,
    initial_customer_impact: initialCustomerImpact,
    system_health: roundTo(initialHealth, 4),
    remaining_budget: task.initial_budget,
    initial_budget: task.initial_budget,
    active_alerts: task.alerts.map((a) => ({ ...a, silenced: false })),
    hidden_root_cause: task.hidden_root_cause,
    root_cause_resolved: false,
    recent_actions: [],
    action_history: [],
    cumulative_reward: 0,
    score: 0,
    done: false,
    success: false,
    cascade_triggered: false,
    reason: 'in_progress',
    services: task.services.map((s) => ({
      ...s,
      inspected: false,
      dependency_inspected: false,
    })),
    max_steps: task.max_steps,
    success_conditions: task.success_conditions,
    task_description: task.description || '',
  };

  stateManager.setState(initialState);

  logger.info(
    `Environment reset: task=${taskName} variant=${initialState.task_variant}`
  );

  const observation = observationBuilder.buildObservation(stateManager.getState());

  return { observation };
}

/**
 * Execute one step in the environment.
 * @param {{ action: string, target?: string }} validatedAction
 * @returns {{ observation, reward, reward_breakdown, done, success, info }}
 */
function step(validatedAction) {
  if (!stateManager.isReady()) {
    const err = new Error('Environment not initialised. Call POST /reset first.');
    err.status = 400;
    throw err;
  }

  const state = stateManager.getState();

  if (state.done) {
    const err = new Error('Episode is already done. Call POST /reset to start a new episode.');
    err.status = 400;
    throw err;
  }

  // Snapshot state BEFORE action (for reward computation)
  const prevState = deepClone(state);

  // 1. Apply the action
  const effects = actionHandler.applyAction(validatedAction);

  // 2. Compute reward
  const { reward, breakdown } = rewardEngine.computeReward(
    prevState,
    validatedAction,
    effects
  );

  // 3. Advance time (passive degradation + customer impact update)
  progressionEngine.advanceStep();

  // 4. Record action in history
  stateManager.appendActionHistory({
    step: prevState.step_count,
    action: validatedAction.action,
    target: validatedAction.target || '',
    reward,
  });

  // 5. Update cumulative reward
  const updatedState = stateManager.getState();
  const cumulative = roundTo(updatedState.cumulative_reward + reward, 4);
  stateManager.patchState({ cumulative_reward: cumulative });

  // 6. Check termination
  const { done, success, reason } = progressionEngine.checkTermination();

  let finalScore = 0;
  let scoreBreakdown = null;

  if (done) {
    stateManager.patchState({ done, success, reason });

    const gradeResult = grader.computeFinalScore();
    finalScore = gradeResult.score;
    scoreBreakdown = gradeResult;

    stateManager.patchState({ score: finalScore });

    logger.info(
      `Episode ended: task=${state.task_name} variant=${state.task_variant || 'v1'} success=${success} score=${finalScore} reason=${reason}`
    );
  }

  // 7. Build observation
  const observation = observationBuilder.buildObservation(stateManager.getState());

  return {
    observation,
    reward,
    reward_breakdown: breakdown,
    cumulative_reward: cumulative,
    done,
    success,
    info: {
      reason,
      step_count: observation.step_count,
      remaining_budget: observation.remaining_budget,
      ...(done
        ? {
            final_score: finalScore,
            score_breakdown: scoreBreakdown,
          }
        : {}),
      effects: effects.effects,
      warnings: effects.warnings,
      task_variant: stateManager.getState().task_variant || 'v1',
    },
  };
}

/**
 * Return the current full internal state snapshot.
 * @returns {object}
 */
function getFullState() {
  if (!stateManager.isReady()) {
    const err = new Error('Environment not initialised. Call POST /reset first.');
    err.status = 400;
    throw err;
  }

  return stateManager.getState();
}

/**
 * Return the current score without ending the episode.
 * @returns {{ score: number, cumulative_reward: number, done: boolean }}
 */
function getCurrentScore() {
  if (!stateManager.isReady()) {
    const err = new Error('Environment not initialised.');
    err.status = 400;
    throw err;
  }

  const state = stateManager.getState();

  return {
    score: state.score,
    cumulative_reward: state.cumulative_reward,
    done: state.done,
    success: state.success,
    step_count: state.step_count,
    task_variant: state.task_variant || 'v1',
  };
}

module.exports = { reset, step, getFullState, getCurrentScore };