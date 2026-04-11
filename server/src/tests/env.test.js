'use strict';

const simulator = require('../engine/simulator');
const stateManager = require('../engine/stateManager');

describe('Environment Core Logic', () => {
  beforeEach(() => {
    stateManager.clearState();
  });

  describe('reset()', () => {
    test('should reset with easy_signal_noise task', () => {
      const { observation } = simulator.reset('easy_signal_noise');
      expect(observation).toBeDefined();
      expect(observation.task_name).toBe('easy_signal_noise');
      expect(observation.step_count).toBe(0);
      expect(observation.services).toHaveLength(6);
      expect(observation.alerts.length).toBeGreaterThan(0);
      expect(observation.remaining_budget).toBe(60);
    });

    test('should reset with medium_hidden_dependency task', () => {
      const { observation } = simulator.reset('medium_hidden_dependency');
      expect(observation.task_name).toBe('medium_hidden_dependency');
      expect(observation.remaining_budget).toBe(75);
    });

    test('should reset with hard_multi_incident task', () => {
      const { observation } = simulator.reset('hard_multi_incident');
      expect(observation.task_name).toBe('hard_multi_incident');
      expect(observation.remaining_budget).toBe(80);
    });

    test('should throw 400 for unknown task', () => {
      expect(() => simulator.reset('nonexistent_task')).toThrow();
    });

    test('observation should not expose hidden_root_cause', () => {
      const { observation } = simulator.reset('easy_signal_noise');
      expect(observation.hidden_root_cause).toBeUndefined();
      expect(observation.root_cause_resolved).toBeUndefined();
    });

    test('internal state should be initialised with all required fields', () => {
      simulator.reset('easy_signal_noise');
      const state = stateManager.getState();
      expect(state.hidden_root_cause).toBe('api_gateway');
      expect(state.root_cause_resolved).toBe(false);
      expect(state.done).toBe(false);
      expect(state.success).toBe(false);
      expect(state.action_history).toHaveLength(0);
    });

    test('second reset should cleanly overwrite previous state', () => {
      simulator.reset('easy_signal_noise');
      simulator.step({ action: 'noop' });
      simulator.reset('medium_hidden_dependency');
      const state = stateManager.getState();
      expect(state.task_name).toBe('medium_hidden_dependency');
      expect(state.step_count).toBe(0);
      expect(state.action_history).toHaveLength(0);
    });
  });

  describe('step()', () => {
    beforeEach(() => simulator.reset('easy_signal_noise'));

    test('should increment step_count', () => {
      simulator.step({ action: 'noop' });
      const state = stateManager.getState();
      expect(state.step_count).toBe(1);
    });

    test('should deduct budget for restart_service', () => {
      const before = stateManager.getState().remaining_budget;
      simulator.step({ action: 'restart_service', target: 'api_gateway' });
      const after = stateManager.getState().remaining_budget;
      expect(after).toBeLessThan(before);
    });

    test('should return reward as a number', () => {
      const result = simulator.step({ action: 'noop' });
      expect(typeof result.reward).toBe('number');
    });

    test('should return done=false when episode is ongoing', () => {
      const result = simulator.step({ action: 'noop' });
      expect(result.done).toBe(false);
    });

    test('should throw if called before reset', () => {
      stateManager.clearState();
      expect(() => simulator.step({ action: 'noop' })).toThrow();
    });

    test('should record action in action_history', () => {
      simulator.step({ action: 'inspect_service', target: 'api_gateway' });
      const state = stateManager.getState();
      expect(state.action_history).toHaveLength(1);
      expect(state.action_history[0].action).toBe('inspect_service');
    });

    test('should set root_cause_resolved when root service health is healed', () => {
      // force-heal the root cause service
      stateManager.updateService('api_gateway', { health: 0.9 });
      simulator.step({ action: 'noop' });
      const state = stateManager.getState();
      expect(state.root_cause_resolved).toBe(true);
    });
  });
});
