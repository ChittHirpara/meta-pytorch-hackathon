'use strict';

const { Router } = require('express');
const ctrl = require('../controllers/envController');

const router = Router();

// OpenEnv interface
router.post('/reset', ctrl.resetEnv);
router.post('/step', ctrl.stepEnv);
router.get('/state', ctrl.getState);
router.get('/tasks', ctrl.getTasks);
router.get('/score', ctrl.getScore);
router.get('/', ctrl.welcome);
router.get('/health', ctrl.healthCheck);

module.exports = router;
