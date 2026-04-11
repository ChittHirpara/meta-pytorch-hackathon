'use strict';

const simulator = require('../engine/simulator');
const stateManager = require('../engine/stateManager');
const { computeFinalScore } = require('../engine/grader');

describe('Grader', () => {
  beforeEach(() => {
    stateManager.clearState();
  });

  test('perfect episode should score close to 1.0', () => {
    simulator.reset('easy_signal_noise');
    // Simulate a good outcome by manually setting state
    stateManager.patchState({
      root_cause_resolved: true,
      customer_impact: 10,
      cascade_triggered: false,
      step_count: 5,
      done: true,
      success: true,
      reason: 'success',
    });
    // Heal all services
    const state = stateManager.getState();
    state.services.forEach((svc) => {
      stateManager.updateService(svc.name, { health: 0.95 });
    });
    stateManager.patchState({ remaining_budget: 50 });

    const result = computeFinalScore();
    expect(result.score).toBeGreaterThan(0.7);
    expect(result.passed).toBe(true);
  });

  test('failed episode (root cause not resolved) should score low', () => {
    simulator.reset('easy_signal_noise');
    stateManager.patchState({
      root_cause_resolved: false,
      customer_impact: 80,
      done: true,
      success: false,
      reason: 'max_steps_reached',
      step_count: 20,
      remaining_budget: 0,
    });

    const result = computeFinalScore();
    expect(result.score).toBeLessThan(0.5);
    expect(result.passed).toBe(false);
    expect(result.root_cause_resolved).toBe(false);
  });

  test('cascade triggered should reduce score', () => {
    simulator.reset('hard_multi_incident');
    stateManager.patchState({
      root_cause_resolved: true,
      customer_impact: 20,
      cascade_triggered: false,
      step_count: 10,
      remaining_budget: 40,
    });
    const state = stateManager.getState();
    state.services.forEach((svc) => stateManager.updateService(svc.name, { health: 0.88 }));
    const scoreNoCascade = computeFinalScore().score;

    stateManager.patchState({ cascade_triggered: true });
    const scoreWithCascade = computeFinalScore().score;

    expect(scoreWithCascade).toBeLessThan(scoreNoCascade);
  });

  test('score breakdown should have all 6 dimensions', () => {
    simulator.reset('easy_signal_noise');
    stateManager.patchState({ done: true, success: false });
    const result = computeFinalScore();
    const bd = result.breakdown;
    expect(bd).toHaveProperty('system_stability');
    expect(bd).toHaveProperty('customer_harm_reduction');
    expect(bd).toHaveProperty('root_cause_resolution');
    expect(bd).toHaveProperty('action_efficiency');
    expect(bd).toHaveProperty('budget_utilisation');
    expect(bd).toHaveProperty('harmful_action_avoidance');
  });

  test('score is always between 0 and 1', () => {
    for (const taskName of ['easy_signal_noise', 'medium_hidden_dependency', 'hard_multi_incident']) {
      simulator.reset(taskName);
      stateManager.patchState({ done: true });
      const result = computeFinalScore();
      expect(result.score).toBeGreaterThanOrEqual(0);
      expect(result.score).toBeLessThanOrEqual(1);
    }
  });

  test('different outcomes should produce different scores (non-constant grader)', () => {
    simulator.reset('easy_signal_noise');
    stateManager.patchState({ root_cause_resolved: true, customer_impact: 10 });
    const state = stateManager.getState();
    state.services.forEach((svc) => stateManager.updateService(svc.name, { health: 0.9 }));
    stateManager.patchState({ remaining_budget: 55 });
    const highScore = computeFinalScore().score;

    simulator.reset('easy_signal_noise');
    stateManager.patchState({ root_cause_resolved: false, customer_impact: 90 });
    const state2 = stateManager.getState();
    state2.services.forEach((svc) => stateManager.updateService(svc.name, { health: 0.2 }));
    stateManager.patchState({ remaining_budget: 5 });
    const lowScore = computeFinalScore().score;

    expect(highScore).toBeGreaterThan(lowScore);
  });
});
