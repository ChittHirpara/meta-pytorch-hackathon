#!/usr/bin/env node
'use strict';

const BASE_URL = process.env.BASE_URL || 'http://localhost:5050';

async function run() {
  console.log('NetPulse Smoke Test\n');

  try {
    const health = await fetch(`${BASE_URL}/health`).then(r => r.json());
    console.log(' Health:', health);
  } catch (e) {
    console.error(' Health check failed:', e.message);
    process.exit(1);
  }

  try {
    const tasks = await fetch(`${BASE_URL}/tasks`).then(r => r.json());
    console.log(' Tasks:', tasks);
  } catch (e) {
    console.error(' Tasks check failed:', e.message);
    process.exit(1);
  }

  try {
    const reset = await fetch(`${BASE_URL}/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task_name: 'easy_signal_noise' }),
    }).then(r => r.json());

    console.log(' Reset OK');

    const step = await fetch(`${BASE_URL}/step`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'inspect_service', target: 'api_gateway' }),
    }).then(r => r.json());

    console.log(' Step OK:', step.reward);
  } catch (e) {
    console.error(' Step test failed:', e.message);
    process.exit(1);
  }

  console.log('\n Smoke test PASSED');
}

run();