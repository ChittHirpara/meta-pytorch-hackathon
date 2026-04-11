#!/usr/bin/env bash

set -e

echo "🚨 TRIAGE-X Local Validation"
echo "===================================="

echo "1. Checking server health..."
curl -s http://localhost:5050/health | jq

echo -e "\n2. Running smoke test..."
node scripts/smoke-test.js

echo -e "\n3. Running test suite..."
npm test

echo -e "\n🔥 All checks passed!"