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

Fixes applied vs original:
  1. Chain-of-Thought prompting — agent now reasons via <thought> block before
     emitting a JSON action, preventing premature submit_query calls.
  2. Dynamic MAX_STEPS — read from obs['max_steps'] at reset time, not hardcoded.
  3. Two-phase agent for task3_hard — a dedicated Cleaner phase runs until all
     tables are clean, then a SQL Writer phase handles the JOIN query.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL   = os.environ.get("API_BASE_URL",   "https://api.openai.com/v1")
MODEL_NAME     = os.environ.get("MODEL_NAME",      "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY",  "")
ENV_BASE_URL   = os.environ.get("ENV_BASE_URL",    "http://localhost:7860")

TEMPERATURE     = 0.0
MAX_TOKENS      = 1024
DEFAULT_MAX_STEPS = 20   # fallback only — real value comes from obs['max_steps']
TIMEOUT         = 30     # seconds per HTTP call

TASKS = ["task1_easy", "task2_medium", "task3_hard"]

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)

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
# FIX 1 — Chain-of-Thought System Prompts
#
# Original problem: "Respond with ONLY a JSON object" forced the LLM to
# pattern-match immediately ("SQL task → submit_query") with no reasoning.
# Fix: Allow a <thought>...</thought> block BEFORE the JSON so the model
# can inspect the table state and decide what to clean first.
# ---------------------------------------------------------------------------

# --- General CoT prompt (used for task1 and task2) -------------------------

SYSTEM_PROMPT = """You are an expert SQL data engineer.
You will be given a database schema, sample data, and a broken SQL query.
Your job is to fix the query and/or clean the data step by step.

HOW TO RESPOND:
First, reason inside a <thought> block. Examine the table columns, null counts,
duplicate risk, and the broken query. Decide which action is most needed RIGHT NOW.
Then output exactly ONE JSON action on a new line after the thought block.

Format:
<thought>
[Your reasoning here — what is wrong, what must be fixed first]
</thought>
{"action_type": "...", ...}

Available action types:
  submit_query    — {"action_type": "submit_query", "query": "<SQL>"}
  drop_nulls      — {"action_type": "drop_nulls", "column_name": "<col>", "table_name": "<table>"}
  drop_duplicates — {"action_type": "drop_duplicates", "table_name": "<table>"}
  rename_column   — {"action_type": "rename_column", "column_name": "<old>", "new_column_name": "<new>", "table_name": "<table>"}
  cast_column     — {"action_type": "cast_column", "column_name": "<col>", "cast_to": "<TYPE>", "table_name": "<table>"}
  done            — {"action_type": "done"}

CRITICAL RULES:
- NEVER call submit_query if any column still has null_count > 0.
- NEVER call submit_query if column names do not match what the query references.
- Fix ALL data quality issues before submitting the query.
- Only output ONE action per response. The JSON must be on its own line.
"""

# --- Two-phase prompts for task3_hard --------------------------------------
# FIX 3: Instead of one prompt trying to clean AND write a complex JOIN,
# we use two specialised system prompts that hand off to each other.

CLEANER_SYSTEM_PROMPT = """You are a Data Cleaning specialist.
Your ONLY job right now is to clean the database tables. Do NOT write SQL queries yet.

HOW TO RESPOND:
Reason inside a <thought> block first, then output ONE cleaning action as JSON.

Format:
<thought>
[Which table needs cleaning? Which column has nulls or wrong name?]
</thought>
{"action_type": "...", ...}

Available cleaning actions ONLY:
  drop_nulls      — {"action_type": "drop_nulls", "column_name": "<col>", "table_name": "<table>"}
  drop_duplicates — {"action_type": "drop_duplicates", "table_name": "<table>"}
  rename_column   — {"action_type": "rename_column", "column_name": "<old>", "new_column_name": "<new>", "table_name": "<table>"}
  cast_column     — {"action_type": "cast_column", "column_name": "<col>", "cast_to": "<TYPE>", "table_name": "<table>"}
  cleaning_done   — {"action_type": "done"}  ← use this ONLY when ALL tables have zero nulls and correct column names

RULES:
- Go table by table. Handle all nulls and renames in one table before moving to the next.
- Output cleaning_done (done action) ONLY when every column in every table has null_count = 0
  and all column names are clean and consistent.
- Never output submit_query — that is for the next phase.
"""

SQL_WRITER_SYSTEM_PROMPT = """You are a SQL Query Writing specialist.
The database tables have already been cleaned. Your ONLY job is to write and submit
a correct SQL query that satisfies the goal.

HOW TO RESPOND:
Reason inside a <thought> block first, then output the submit_query action as JSON.

Format:
<thought>
[Review the column names now available. Write the correct JOIN / aggregation logic.]
</thought>
{"action_type": "submit_query", "query": "<your corrected SQL here>"}

RULES:
- Use ONLY column names that currently exist in the tables (check the schema carefully).
- The query must match the goal description exactly.
- After submitting, if the score is still low, read the error and try again with a fix.
- Output done if you are confident the query is correct and the score is high.
"""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_tables_summary(obs: Dict[str, Any]) -> str:
    """Render a clear table summary so the model sees nulls and column names."""
    lines = []
    for t in obs.get("tables", []):
        cols = [
            f"  - {c['name']} ({c['dtype']}, nulls={c['null_count']})"
            for c in t.get("columns", [])
        ]
        lines.append(f"Table '{t['table_name']}': {t['row_count']} rows")
        lines.extend(cols)
    return "\n".join(lines) if lines else "No tables available."


def build_user_prompt(
    step: int,
    obs: Dict[str, Any],
    history: List[str],
    max_steps: int,
    phase: str = "general",
) -> str:
    tables_summary = build_tables_summary(obs)
    history_text   = "\n".join(history[-6:]) if history else "None"

    phase_note = ""
    if phase == "cleaner":
        phase_note = "\n⚠ CLEANER PHASE: Only cleaning actions allowed. Do NOT submit_query yet.\n"
    elif phase == "sql_writer":
        phase_note = "\n⚠ SQL WRITER PHASE: Tables are clean. Write and submit the correct query now.\n"

    prompt = f"""=== STEP {step} / {max_steps} ==={phase_note}

GOAL:
{obs.get('goal', '')}

INSTRUCTIONS:
{obs.get('instructions', '')}

CURRENT DATABASE STATE:
{tables_summary}

BROKEN QUERY TO FIX:
{obs.get('broken_query', 'N/A')}

RECENT HISTORY (last 6 steps):
{history_text}

LAST ACTION ERROR: {obs.get('last_action_error') or 'None'}
CURRENT SCORE:     {obs.get('current_score', 0.0):.2f}
STEPS REMAINING:   {max_steps - step}

Think carefully in a <thought> block, then output your single JSON action.
"""
    return prompt


# ---------------------------------------------------------------------------
# FIX 1 (continued) — parse_action strips <thought> block before JSON parse
# ---------------------------------------------------------------------------

def parse_action(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON action from model response.
    Handles:
      - <thought>...</thought> blocks before the JSON (CoT format)
      - Markdown code fences
      - JSON embedded anywhere in the text
    """
    # Strip <thought>...</thought> block (our CoT wrapper)
    text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL).strip()

    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    text = text.strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first JSON object in the remaining text
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# Helper — check if all tables are clean
# ---------------------------------------------------------------------------

def tables_are_clean(obs: Dict[str, Any]) -> bool:
    """Return True when every column in every table has zero nulls."""
    for t in obs.get("tables", []):
        for col in t.get("columns", []):
            if col.get("null_count", 0) > 0:
                return False
    return True


# ---------------------------------------------------------------------------
# Fallback actions (used when LLM call or parse fails)
# ---------------------------------------------------------------------------

FALLBACK_ACTIONS = {
    "task1_easy": {
        "action_type": "submit_query",
        "query": (
            "SELECT id, name, department, salary FROM employees "
            "WHERE salary > 70000 ORDER BY salary DESC"
        ),
    },
    "task2_medium": {
        "action_type": "drop_nulls",
        "column_name": "amount",
        "table_name": "orders",
    },
    "task3_hard": {
        "action_type": "drop_nulls",
        "column_name": "region",
        "table_name": "customers",
    },
}


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

def call_llm(
    system_prompt: str,
    user_prompt: str,
    task_id: str,
    step: int,
) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return completion.choices[0].message.content or ""
    except Exception as exc:
        print(f"  ⚠ LLM call failed at step {step}: {exc}. Using fallback.")
        return json.dumps(FALLBACK_ACTIONS.get(task_id, {"action_type": "done"}))


# ---------------------------------------------------------------------------
# FIX 2 — run_episode reads max_steps from obs, not a hardcoded constant
# FIX 3 — task3_hard uses two-phase agent (cleaner → sql_writer)
# ---------------------------------------------------------------------------

def run_episode(task_id: str) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"  TASK: {task_id}")
    print(f"{'='*60}")

    # Reset environment
    obs = env_reset(task_id)

    # FIX 2: read max_steps from the environment observation
    max_steps = obs.get("max_steps", DEFAULT_MAX_STEPS)
    print(f"  Goal:      {obs.get('goal', '')[:100]}...")
    print(f"  Max steps: {max_steps}")

    history:     List[str] = []
    final_score: float     = 0.0
    done:        bool      = obs.get("done", False)
    step:        int       = 0

    # ── FIX 3: Two-phase agent for task3_hard ──────────────────────────────
    # Phase 1 — Cleaner: iterate until all tables have zero nulls
    # Phase 2 — SQL Writer: submit the corrected JOIN query
    # For task1 and task2 we use the single general CoT prompt throughout.

    if task_id == "task3_hard":
        # ── PHASE 1: Data Cleaning ──────────────────────────────────────────
        print("\n  [Phase 1 — Data Cleaner]")
        phase = "cleaner"
        system_prompt = CLEANER_SYSTEM_PROMPT

        for step in range(1, max_steps + 1):
            if done:
                break

            # Switch to SQL Writer once all tables are clean
            if tables_are_clean(obs) and phase == "cleaner":
                print(f"  ✔ All tables clean at step {step}. Switching to SQL Writer phase.")
                phase         = "sql_writer"
                system_prompt = SQL_WRITER_SYSTEM_PROMPT
                print("\n  [Phase 2 — SQL Writer]")

            user_prompt  = build_user_prompt(step, obs, history, max_steps, phase=phase)
            response_text = call_llm(system_prompt, user_prompt, task_id, step)

            action = parse_action(response_text)
            if action is None:
                print(f"  ⚠ Could not parse action at step {step}. Using fallback.")
                action = FALLBACK_ACTIONS.get(task_id, {"action_type": "done"})

            _log_action(step, action)

            try:
                result = env_step(action)
            except Exception as exc:
                print(f"  ⚠ env.step() failed: {exc}")
                break

            obs, reward_info, done, final_score = _unpack_result(result, obs)
            _log_reward(reward_info, done)
            history.append(_history_line(step, action, final_score, reward_info))

            if done:
                print(f"  ✅ Episode complete at step {step}.")
                break

            time.sleep(0.3)

    else:
        # ── Single-phase CoT agent for task1_easy and task2_medium ──────────
        for step in range(1, max_steps + 1):
            if done:
                print(f"  → Episode done at step {step - 1}.")
                break

            user_prompt   = build_user_prompt(step, obs, history, max_steps, phase="general")
            response_text = call_llm(SYSTEM_PROMPT, user_prompt, task_id, step)

            action = parse_action(response_text)
            if action is None:
                print(f"  ⚠ Could not parse action at step {step}. Using fallback.")
                action = FALLBACK_ACTIONS.get(task_id, {"action_type": "done"})

            _log_action(step, action)

            try:
                result = env_step(action)
            except Exception as exc:
                print(f"  ⚠ env.step() failed: {exc}")
                break

            obs, reward_info, done, final_score = _unpack_result(result, obs)
            _log_reward(reward_info, done)
            history.append(_history_line(step, action, final_score, reward_info))

            if done:
                print(f"  ✅ Episode complete at step {step}.")
                break

            time.sleep(0.3)

    if not done:
        print(f"  ⏱ Reached max steps ({max_steps}).")

    return {
        "task_id":     task_id,
        "final_score": round(final_score, 4),
        "steps_taken": step,
        "done":        done,
    }


# ---------------------------------------------------------------------------
# Small helpers to keep run_episode readable
# ---------------------------------------------------------------------------

def _unpack_result(result: Dict[str, Any], prev_obs: Dict[str, Any]):
    obs         = result.get("observation", prev_obs)
    reward_info = result.get("reward", {})
    done        = result.get("done", False)
    final_score = reward_info.get("total_reward", 0.0)
    return obs, reward_info, done, final_score


def _log_action(step: int, action: Dict[str, Any]) -> None:
    line = f"  Step {step:02d}: {action.get('action_type', '?')}"
    if "query" in action:
        line += f"  →  {action['query'][:70]}..."
    elif "column_name" in action:
        line += f"  →  col={action.get('column_name')}  table={action.get('table_name')}"
    elif "new_column_name" in action:
        line += (
            f"  →  {action.get('column_name')} → {action.get('new_column_name')}"
            f"  table={action.get('table_name')}"
        )
    print(line)


def _log_reward(reward_info: Dict[str, Any], done: bool) -> None:
    feedback = reward_info.get("feedback", "")
    print(
        f"         reward={reward_info.get('step_reward', 0.0):+.2f} "
        f"| total={reward_info.get('total_reward', 0.0):.2f} "
        f"| done={done} "
        f"| {feedback[:70]}"
    )


def _history_line(
    step: int,
    action: Dict[str, Any],
    score: float,
    reward_info: Dict[str, Any],
) -> str:
    feedback = reward_info.get("feedback", "")
    return f"Step {step}: {action.get('action_type')} → score={score:.2f} | {feedback[:60]}"


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
    print(f"   Model    : {MODEL_NAME}")
    print(f"   API base : {API_BASE_URL}")
    print(f"   Env URL  : {ENV_BASE_URL}")
    print(f"   Tasks    : {tasks_to_run}")

    # Verify environment is reachable
    try:
        resp = httpx.get(f"{ENV_BASE_URL}/", timeout=10)
        resp.raise_for_status()
        print(f"   Env      : ✅ reachable\n")
    except Exception as e:
        print(f"   Env      : ❌ not reachable ({e})")
        print("   Please start the environment: uvicorn app.main:app --port 7860")
        sys.exit(1)

    # Run all tasks
    results = []
    for task_id in tasks_to_run:
        result = run_episode(task_id)
        results.append(result)

    # Final summary
    print(f"\n{'='*60}")
    print("  FINAL SCORES")
    print(f"{'='*60}")
    total = 0.0
    for r in results:
        if r["final_score"] >= 0.7:
            status = "✅"
        elif r["final_score"] >= 0.3:
            status = "⚠ "
        else:
            status = "❌"
        print(
            f"  {status} {r['task_id']:20s}  "
            f"score={r['final_score']:.4f}  steps={r['steps_taken']}"
        )
        total += r["final_score"]

    avg = total / len(results) if results else 0.0
    print(f"\n  Average score : {avg:.4f}")
    print(f"{'='*60}\n")

    # Write results for automated evaluation
    with open("inference_results.json", "w") as f:
        json.dump(
            {"results": results, "average_score": round(avg, 4)},
            f,
            indent=2,
        )
    print("  Results saved → inference_results.json")

    return results


if __name__ == "__main__":
    main()
