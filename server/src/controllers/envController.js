'use strict';

const simulator = require('../engine/simulator');
const { validateAction } = require('../engine/validationEngine');
const { listTasks } = require('../engine/taskLoader');
const { success, error } = require('../utils/responseBuilder');
const logger = require('../utils/logger');

async function resetEnv(req, res, next) {
  try {
    const { task_name } = req.body;
    // OpenEnv 1.0 specification allows reset without task_name (defaults to first task)
    const activeTask = task_name || 'easy_signal_noise';

    if (typeof activeTask !== 'string') {
      return res.status(400).json(error('task_name must be a string if provided.'));
    }
    
    const result = simulator.reset(activeTask.trim());
    return res.status(200).json(success(result));
  } catch (err) {
    next(err);
  }
}


async function stepEnv(req, res, next) {
  try {
    const validatedAction = validateAction(req.body);
    const result = simulator.step(validatedAction);
    return res.status(200).json(success(result));
  } catch (err) {
    next(err);
  }
}


async function getState(req, res, next) {
  try {
    const state = simulator.getFullState();
    return res.status(200).json(success({ state }));
  } catch (err) {
    next(err);
  }
}


async function getTasks(req, res, next) {
  try {
    const tasks = listTasks();
    return res.status(200).json(success({ tasks }));
  } catch (err) {
    next(err);
  }
}


async function getScore(req, res, next) {
  try {
    const scoreData = simulator.getCurrentScore();
    return res.status(200).json(success(scoreData));
  } catch (err) {
    next(err);
  }
}

async function healthCheck(req, res) {
  return res.status(200).json({
    ok: true,
    service: 'netpulse-server',
    version: '1.0.0',
    timestamp: new Date().toISOString(),
  });
}

async function welcome(req, res) {
  return res.status(200).json({
    message: 'Welcome to the NetPulse Incident Response Benchmark Engine.',
    spec: 'OpenEnv 1.0 Compliant',
    author: 'Chitt Hirpara',
    endpoints: {
      reset: 'POST /reset',
      step: 'POST /step',
      state: 'GET /state',
      tasks: 'GET /tasks',
      score: 'GET /score',
      health: 'GET /health',
    },
    documentation: 'https://github.com/ChittHirpara/meta-pytorch-hackathon.git',
  });
}

module.exports = { resetEnv, stepEnv, getState, getTasks, getScore, healthCheck, welcome };
