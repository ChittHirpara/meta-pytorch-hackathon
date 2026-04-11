#!/usr/bin/env node
'use strict';

const BASE_URL = process.env.BASE_URL || 'http://localhost:5050';

const TASKS = [
  'easy_signal_noise',
  'medium_hidden_dependency',
  'hard_multi_incident',
];

async function post(path, body) {
  return fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(r => r.json());
}

async function get(path) {
  return fetch(`${BASE_URL}${path}`).then(r => r.json());
}

async function runTask(task) {
  const reset = await post('/reset', { task_name: task });
  let obs = reset.observation;

  let steps = 0;

  while (steps < 20) {
    const action = { action: 'noop' };

    const res = await post('/step', action);
    obs = res.observation;

    if (res.done) {
      const score = await get('/score');
      return {
        task,
        success: res.success,
        score: score.score,
        steps: obs.step_count,
      };
    }

    steps++;
  }

  return { task, success: false, score: 0 };
}

async function main() {
  console.log('📊 Generating NetPulse Report\n');

  const results = [];

  for (const task of TASKS) {
    console.log(`Running ${task}...`);
    const res = await runTask(task);
    results.push(res);
  }

  const avg =
    results.reduce((sum, r) => sum + (r.score || 0), 0) / results.length;

  console.log('\n📈 RESULTS:');
  console.table(results);

  console.log(`\n🏆 Average Score: ${avg.toFixed(4)}`);
}

main();