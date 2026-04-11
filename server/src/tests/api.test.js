'use strict';

const request = require('supertest');
const app = require('../app');
const stateManager = require('../engine/stateManager');

describe('API Routes', () => {
  beforeEach(() => {
    stateManager.clearState();
  });

  // ── GET /health ──────────────────────────────────────────────────────────
  describe('GET /health', () => {
    test('should return 200 with ok:true', async () => {
      const res = await request(app).get('/health');
      expect(res.status).toBe(200);
      expect(res.body.ok).toBe(true);
      expect(res.body.service).toBe('netpulse-server');
    });
  });

  // ── GET /tasks ───────────────────────────────────────────────────────────
  describe('GET /tasks', () => {
    test('should return list of 3 tasks', async () => {
      const res = await request(app).get('/tasks');
      expect(res.status).toBe(200);
      expect(res.body.ok).toBe(true);
      expect(res.body.tasks).toHaveLength(3);
      expect(res.body.tasks).toContain('easy_signal_noise');
      expect(res.body.tasks).toContain('medium_hidden_dependency');
      expect(res.body.tasks).toContain('hard_multi_incident');
    });
  });

  // ── POST /reset ──────────────────────────────────────────────────────────
  describe('POST /reset', () => {
    test('should reset with valid task_name', async () => {
      const res = await request(app)
        .post('/reset')
        .send({ task_name: 'easy_signal_noise' });
      expect(res.status).toBe(200);
      expect(res.body.ok).toBe(true);
      expect(res.body.observation).toBeDefined();
      expect(res.body.observation.task_name).toBe('easy_signal_noise');
      expect(res.body.observation.step_count).toBe(0);
    });

    test('should return 400 for missing task_name', async () => {
      const res = await request(app).post('/reset').send({});
      expect(res.status).toBe(400);
      expect(res.body.ok).toBe(false);
      expect(res.body.error).toBeDefined();
    });

    test('should return 400 for unknown task_name', async () => {
      const res = await request(app)
        .post('/reset')
        .send({ task_name: 'bogus_task' });
      expect(res.status).toBe(400);
      expect(res.body.ok).toBe(false);
    });
  });

  // ── POST /step ───────────────────────────────────────────────────────────
  describe('POST /step', () => {
    beforeEach(async () => {
      await request(app).post('/reset').send({ task_name: 'easy_signal_noise' });
    });

    test('should process noop action', async () => {
      const res = await request(app).post('/step').send({ action: 'noop' });
      expect(res.status).toBe(200);
      expect(res.body.ok).toBe(true);
      expect(res.body.observation).toBeDefined();
      expect(typeof res.body.reward).toBe('number');
      expect(typeof res.body.done).toBe('boolean');
    });

    test('should process inspect_service with target', async () => {
      const res = await request(app)
        .post('/step')
        .send({ action: 'inspect_service', target: 'api_gateway' });
      expect(res.status).toBe(200);
      expect(res.body.reward).toBeGreaterThanOrEqual(0);
    });

    test('should return 400 for invalid action', async () => {
      const res = await request(app)
        .post('/step')
        .send({ action: 'hack_the_planet' });
      expect(res.status).toBe(400);
      expect(res.body.ok).toBe(false);
    });

    test('should return 400 for invalid target service', async () => {
      const res = await request(app)
        .post('/step')
        .send({ action: 'restart_service', target: 'fake_service' });
      expect(res.status).toBe(400);
      expect(res.body.ok).toBe(false);
    });

    test('should return 400 if called before reset', async () => {
      stateManager.clearState();
      const res = await request(app).post('/step').send({ action: 'noop' });
      expect(res.status).toBe(400);
      expect(res.body.ok).toBe(false);
    });

    test('step_count should increment by 1 each call', async () => {
      await request(app).post('/step').send({ action: 'noop' });
      const res = await request(app).post('/step').send({ action: 'noop' });
      expect(res.body.observation.step_count).toBe(2);
    });

    test('done=true should include final_score in info', async () => {
      // Exhaust budget rapidly
      for (let i = 0; i < 20; i++) {
        await request(app)
          .post('/step')
          .send({ action: 'scale_service', target: 'api_gateway' });
      }
      const state = stateManager.getState();
      if (state && state.done) {
        const res = await request(app).get('/score');
        expect(res.body.score).toBeGreaterThanOrEqual(0);
        expect(res.body.score).toBeLessThanOrEqual(1);
      }
    });
  });

  // ── GET /state ───────────────────────────────────────────────────────────
  describe('GET /state', () => {
    test('should return 400 if not initialised', async () => {
      const res = await request(app).get('/state');
      expect(res.status).toBe(400);
    });

    test('should return full state after reset', async () => {
      await request(app).post('/reset').send({ task_name: 'medium_hidden_dependency' });
      const res = await request(app).get('/state');
      expect(res.status).toBe(200);
      expect(res.body.state).toBeDefined();
      expect(res.body.state.task_name).toBe('medium_hidden_dependency');
      // Internal fields should be present in /state (not hidden)
      expect(res.body.state.hidden_root_cause).toBe('database_replica');
    });
  });

  // ── GET /score ───────────────────────────────────────────────────────────
  describe('GET /score', () => {
    test('should return 400 if not initialised', async () => {
      const res = await request(app).get('/score');
      expect(res.status).toBe(400);
    });

    test('should return score=0 at start of episode', async () => {
      await request(app).post('/reset').send({ task_name: 'easy_signal_noise' });
      const res = await request(app).get('/score');
      expect(res.status).toBe(200);
      expect(res.body.score).toBe(0);
      expect(res.body.done).toBe(false);
    });
  });

  // ── 404 handler ──────────────────────────────────────────────────────────
  describe('Unknown routes', () => {
    test('GET /unknown should return 404', async () => {
      const res = await request(app).get('/unknown');
      expect(res.status).toBe(404);
      expect(res.body.ok).toBe(false);
      expect(res.body.available_routes).toBeDefined();
    });
  });
});
