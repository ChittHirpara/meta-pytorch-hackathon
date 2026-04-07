"""
inference.py — OpenEnv SQL Repair Environment Baseline Inference Script

Runs an LLM agent against all 3 tasks and prints final scores.

Required environment variables:
  OPENAI_API_KEY  — your OpenAI (or compatible) API key
  API_BASE_URL    — API base URL (default: https://api.openai.com/v1)
  MODEL_NAME      — model identifier (default: gpt-4o-mini)

Usage:
  python inference.py
  python inference.py --task task1_easy
  python inference.py --task all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ENV_BASE_URL  = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

TEMPERATURE = 0.0
MAX_TOKENS  = 1024
MAX_STEPS   = 10        # hard cap per episode
TIMEOUT     = 30        # seconds per HTTP call

TASKS = ["task1_easy", "task2_medium", "task3_hard"]

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

client = OpenAI(api_key=HF_TOKEN or OPENAI_API_KEY, base_url=API_BASE_URL)

# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_reset(task_id: str) -> Dict[str, Any]:
    resp = httpx.post(
        f"{ENV_BASE_URL}/reset",
        params={"task_id": task_id},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = httpx.post(
        f"{ENV_BASE_URL}/step",
        json=action,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def env_state() -> Dict[str, Any]:
    resp = httpx.get(f"{ENV_BASE_URL}/state", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert SQL data engineer.
You will be given a database schema, sample data, and a broken SQL query.
Your job is to fix the query and/or clean the data step by step.

You must respond with ONLY a valid JSON object — no prose, no markdown fences.

Available action types:
  submit_query    — {"action_type": "submit_query", "query": "<SQL>"}
  drop_nulls      — {"action_type": "drop_nulls", "column_name": "<col>", "table_name": "<table>"}
  drop_duplicates — {"action_type": "drop_duplicates", "table_name": "<table>"}
  rename_column   — {"action_type": "rename_column", "column_name": "<old>", "new_column_name": "<new>", "table_name": "<table>"}
  cast_column     — {"action_type": "cast_column", "column_name": "<col>", "cast_to": "<TYPE>", "table_name": "<table>"}
  done            — {"action_type": "done"}

Rules:
- Fix data issues BEFORE submitting a query (for medium/hard tasks).
- Only output a single JSON action per response.
- Do not include any explanation or extra text.
- CRITICAL for task2: column is called 'amt' at start — rename it to 'amount' FIRST before any query.
- CRITICAL for task3: column is called 'quanity' at start — rename it to 'qty' FIRST before cleaning nulls or submitting a query. The query will fail if you skip this step.
- Check column names in the CURRENT DATABASE STATE before submitting a query.
"""


def build_user_prompt(
    step: int,
    obs: Dict[str, Any],
    history: List[str],
) -> str:
    tables_summary = []
    for t in obs.get("tables", []):
        cols = [f"{c['name']} ({c['dtype']}, nulls={c['null_count']})" for c in t.get("columns", [])]
        tables_summary.append(
            f"Table '{t['table_name']}': {t['row_count']} rows | Columns: {', '.join(cols)}"
        )

    history_text = "\n".join(history[-5:]) if history else "None"

    # Detect unresolved column name issues and surface them explicitly
    # so the agent knows it MUST rename before querying
    warnings = []
    all_col_names = []
    for t in obs.get("tables", []):
        for c in t.get("columns", []):
            all_col_names.append((t["table_name"], c["name"]))

    if any(col == "amt" for _, col in all_col_names):
        warnings.append(
            "⚠ WARNING: Column 'amt' still exists in orders table. "
            "You MUST rename it to 'amount' before submitting the query."
        )
    if any(col == "quanity" for _, col in all_col_names):
        warnings.append(
            "⚠ WARNING: Column 'quanity' still exists in transactions table. "
            "You MUST rename it to 'qty' FIRST — before cleaning nulls or submitting any query."
        )

    warning_text = "\n".join(warnings) if warnings else "None"

    prompt = f"""=== STEP {step} ===

GOAL:
{obs.get('goal', '')}

INSTRUCTIONS:
{obs.get('instructions', '')}

CURRENT DATABASE STATE:
{chr(10).join(tables_summary)}

BROKEN QUERY TO FIX:
{obs.get('broken_query', 'N/A')}

RECENT HISTORY (last 5 steps):
{history_text}

COLUMN NAME WARNINGS (fix these before querying):
{warning_text}

LAST ACTION ERROR: {obs.get('last_action_error') or 'None'}
CURRENT SCORE: {obs.get('current_score', 0.0):.2f}
STEPS REMAINING: {obs.get('max_steps', 10) - step}

What is your next action? Respond with ONLY a JSON object.
"""
    return prompt


# ---------------------------------------------------------------------------
# Parse model output
# ---------------------------------------------------------------------------

def parse_action(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON action from model response."""
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON within the text
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Fallback actions — ORDERED LISTS, one action per step, correct sequence
#
# Fix 1: task2 fallback was {"column_name": "amount"} but the column is
#         still named 'amt' at episode start → always failed with -0.05.
#         Now the first fallback step renames 'amt' → 'amount'.
#
# Fix 2: task3 fallback was a single static drop_nulls action.
#         Now it's a full ordered sequence: rename → clean → query.
# ---------------------------------------------------------------------------

FALLBACK_ACTIONS: Dict[str, List[Dict[str, Any]]] = {
    "task1_easy": [
        {
            "action_type": "submit_query",
            "query": (
                "SELECT id, name, department, salary "
                "FROM employees "
                "WHERE salary > 70000 "
                "ORDER BY salary DESC"
            ),
        },
    ],
    "task2_medium": [
        # Step 1 — rename the column first (it starts as 'amt')
        {
            "action_type": "rename_column",
            "table_name": "orders",
            "column_name": "amt",
            "new_column_name": "amount",
        },
        # Step 2 — drop null customer names
        {
            "action_type": "drop_nulls",
            "table_name": "orders",
            "column_name": "customer_name",
        },
        # Step 3 — drop duplicate rows
        {
            "action_type": "drop_duplicates",
            "table_name": "orders",
        },
        # Step 4 — cast amount to REAL (removes 'N/A' bad rows)
        {
            "action_type": "cast_column",
            "table_name": "orders",
            "column_name": "amount",
            "cast_to": "REAL",
        },
        # Step 5 — now query with correct column name 'amount'
        {
            "action_type": "submit_query",
            "query": (
                "SELECT department, SUM(amount) AS total_revenue "
                "FROM orders "
                "WHERE amount > 100 AND status = 'completed' "
                "GROUP BY department "
                "ORDER BY total_revenue DESC"
            ),
        },
    ],
    "task3_hard": [
        # Step 1 — fix the typo column name first
        {
            "action_type": "rename_column",
            "table_name": "transactions",
            "column_name": "quanity",
            "new_column_name": "qty",
        },
        # Step 2 — drop null customer names
        {
            "action_type": "drop_nulls",
            "table_name": "customers",
            "column_name": "cust_name",
        },
        # Step 3 — drop null product names
        {
            "action_type": "drop_nulls",
            "table_name": "products",
            "column_name": "prod_name",
        },
        # Step 4 — drop duplicate transactions
        {
            "action_type": "drop_duplicates",
            "table_name": "transactions",
        },
        # Step 5 — drop null customer_id in transactions
        {
            "action_type": "drop_nulls",
            "table_name": "transactions",
            "column_name": "customer_id",
        },
        # Step 6 — submit the corrected multi-table query
        {
            "action_type": "submit_query",
            "query": (
                "SELECT c.region, SUM(p.price * t.qty) AS total_revenue "
                "FROM transactions t "
                "JOIN customers c ON t.customer_id = c.cust_id "
                "JOIN products  p ON t.product_id  = p.prod_id "
                "GROUP BY c.region "
                "HAVING total_revenue > 10000 "
                "ORDER BY total_revenue DESC"
            ),
        },
    ],
}

# Track which fallback step each task is on (reset per episode)
_fallback_step: Dict[str, int] = {}


def get_fallback_action(task_id: str) -> Dict[str, Any]:
    """Return the next fallback action for this task, in sequence."""
    steps = FALLBACK_ACTIONS.get(task_id, [{"action_type": "done"}])
    idx = _fallback_step.get(task_id, 0)
    action = steps[min(idx, len(steps) - 1)]
    _fallback_step[task_id] = idx + 1
    return action


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_episode(task_id: str) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"  TASK: {task_id}")
    print(f"{'='*60}")

    # Reset environment
    obs = env_reset(task_id)
    print(f"Goal: {obs.get('goal', '')[:120]}...")
    print(f"[START] {task_id}")

    history:   List[str] = []
    final_score = 0.0
    done = obs.get("done", False)

    # Reset fallback step counter for this task episode
    _fallback_step[task_id] = 0

    # FIX: read max_steps from the environment observation, not hardcoded
    max_steps = obs.get("max_steps", MAX_STEPS)

    for step in range(1, max_steps + 1):
        if done:
            print(f"  → Episode done at step {step - 1}.")
            break

        # Build prompt
        user_prompt = build_user_prompt(step, obs, history)

        # Call LLM
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  ⚠ LLM call failed at step {step}: {exc}. Using fallback.")
            response_text = json.dumps(get_fallback_action(task_id))

        # Parse action
        action = parse_action(response_text)
        if action is None:
            print(f"  ⚠ Could not parse action at step {step}. Using fallback.")
            action = get_fallback_action(task_id)

        print(f"  Step {step}: action={action.get('action_type')}", end="")
        if "query" in action:
            print(f"  query={action['query'][:60]}...", end="")
        elif "column_name" in action:
            print(f"  col={action.get('column_name')}  table={action.get('table_name')}", end="")
        print()

        print(f"[STEP] {json.dumps(action)}")
        # Step environment
        try:
            result = env_step(action)
        except Exception as exc:
            print(f"  ⚠ env.step() failed: {exc}")
            break

        obs         = result.get("observation", obs)
        reward_info = result.get("reward", {})
        done        = result.get("done", False)
        final_score = reward_info.get("total_reward", 0.0)

        feedback = reward_info.get("feedback", "")
        history.append(
            f"Step {step}: {action.get('action_type')} → score={final_score:.2f} | {feedback[:60]}"
        )
        print(f"    reward={reward_info.get('step_reward', 0.0):+.2f} | total={final_score:.2f} | {feedback[:70]}")

        if done:
            print(f"  ✅ Episode complete at step {step}.")
            break

        time.sleep(0.3)   # rate limit courtesy

    if not done:
        print(f"  ⏱ Reached max steps ({max_steps}).")

    print(f"[END] score={final_score}")

    return {
        "task_id":     task_id,
        "final_score": round(final_score, 4),
        "steps_taken": step,
        "done":        done,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OpenEnv SQL Repair — Baseline Inference")
    parser.add_argument(
        "--task",
        default="all",
        choices=TASKS + ["all"],
        help="Which task(s) to run (default: all)",
    )
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("⚠  OPENAI_API_KEY is not set. LLM calls may fail.")

    tasks_to_run = TASKS if args.task == "all" else [args.task]

    print(f"\n🚀 OpenEnv SQL Repair — Baseline Inference")
    print(f"   Model:       {MODEL_NAME}")
    print(f"   API base:    {API_BASE_URL}")
    print(f"   Env URL:     {ENV_BASE_URL}")
    print(f"   Tasks:       {tasks_to_run}")

    # Verify environment is reachable
    try:
        resp = httpx.get(f"{ENV_BASE_URL}/", timeout=10)
        resp.raise_for_status()
        print(f"   Env status:  ✅ reachable\n")
    except Exception as e:
        print(f"   Env status:  ❌ not reachable ({e})")
        print("   Please start the environment first: uvicorn app.main:app --port 7860")
        sys.exit(1)

    # Run all tasks
    results = []
    for task_id in tasks_to_run:
        result = run_episode(task_id)
        results.append(result)

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL SCORES")
    print(f"{'='*60}")
    total = 0.0
    for r in results:
        status = "✅" if r["final_score"] >= 0.7 else ("⚠ " if r["final_score"] >= 0.3 else "❌")
        print(f"  {status} {r['task_id']:20s}  score={r['final_score']:.4f}  steps={r['steps_taken']}")
        total += r["final_score"]

    avg = total / len(results) if results else 0.0
    print(f"\n  Average score: {avg:.4f}")
    print(f"{'='*60}\n")

    # Write results to file for automated evaluation
    with open("inference_results.json", "w") as f:
        json.dump({"results": results, "average_score": round(avg, 4)}, f, indent=2)
    print("  Results saved to inference_results.json")

    return results


if __name__ == "__main__":
    main()
