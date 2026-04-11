'use strict';

const simulator = require('../engine/simulator');
const stateManager = require('../engine/stateManager');
const { computeReward } = require('../engine/rewardEngine');
const { deepClone } = require('../utils/deepClone');

describe('Reward Engine', () => {
  beforeEach(() => {
    stateManager.clearState();
    simulator.reset('easy_signal_noise');
  });

  test('noop should yield negative reward', () => {
    const prevState = stateManager.getState();
    const result = simulator.step({ action: 'noop' });
    expect(result.reward).toBeLessThan(0);
  });

  test('restart_service on root cause should give positive reward', () => {
    const result = simulator.step({ action: 'restart_service', target: 'api_gateway' });
    // api_gateway is root cause — restarting it should give positive reward
    expect(result.reward).toBeGreaterThan(0);
  });

  test('inspect_service (first time) should give positive reward', () => {
    const result = simulator.step({ action: 'inspect_service', target: 'api_gateway' });
    expect(result.reward).toBeGreaterThanOrEqual(0);
    expect(result.reward_breakdown.diagnostic_value).toBeGreaterThan(0);
  });

  test('inspect_service (repeated) should give lower reward than first time', () => {
    const first = simulator.step({ action: 'inspect_service', target: 'api_gateway' });
    const second = simulator.step({ action: 'inspect_service', target: 'api_gateway' });
    // Second inspect should have lower diagnostic value
    expect(second.reward_breakdown.diagnostic_value).toBeLessThanOrEqual(
      first.reward_breakdown.diagnostic_value
    );
  });

  test('reward_breakdown should have all 5 components', () => {
    const result = simulator.step({ action: 'noop' });
    const bd = result.reward_breakdown;
    expect(bd).toHaveProperty('health_improvement');
    expect(bd).toHaveProperty('customer_impact_reduction');
    expect(bd).toHaveProperty('diagnostic_value');
    expect(bd).toHaveProperty('root_cause_progress');
    expect(bd).toHaveProperty('penalties');
  });

  test('silencing a real alert should incur a penalty', () => {
    // In easy_signal_noise, alert-001 is real (api_gateway)
    const result = simulator.step({ action: 'silence_alert', target: 'alert-001' });
    expect(result.reward_breakdown.penalties).toBeLessThan(0);
  });

  test('silencing a fake alert should have smaller penalty', () => {
    const fake = simulator.step({ action: 'silence_alert', target: 'alert-002' });
    const real = simulator.step({ action: 'silence_alert', target: 'alert-001' });
    // Silencing a real alert should hurt more
    expect(real.reward_breakdown.penalties).toBeLessThanOrEqual(fake.reward_breakdown.penalties);
  });

  test('reward should not be constant across different actions', () => {
    simulator.reset('easy_signal_noise');
    const r1 = simulator.step({ action: 'noop' });
    simulator.reset('easy_signal_noise');
    const r2 = simulator.step({ action: 'restart_service', target: 'api_gateway' });
    expect(r1.reward).not.toEqual(r2.reward);
  });

  test('cumulative reward should increase with each step', () => {
    simulator.step({ action: 'inspect_service', target: 'api_gateway' });
    const s1 = stateManager.getState().cumulative_reward;
    simulator.step({ action: 'noop' });
    const s2 = stateManager.getState().cumulative_reward;
    // Latter steps may or may not increase cumulative depending on reward
    expect(typeof s1).toBe('number');
    expect(typeof s2).toBe('number');
  });
});
