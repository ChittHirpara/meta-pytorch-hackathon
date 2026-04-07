"""
Grader — Task 1 (Easy)
Scores the agent's submitted query against the ground truth output.
"""

import sqlite3
from typing import Any, Dict, List

from app.tasks.task1_easy import EXPECTED_ROWS, run_query


def _normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {k: (v.strip().lower() if isinstance(v, str) else v) for k, v in row.items()}
        for row in rows
    ]


def grade(
    conn: sqlite3.Connection,
    submitted_query: str,
    step_count: int,
    max_steps: int,
) -> Dict[str, Any]:
    success, rows, error = run_query(conn, submitted_query)
    if not success:
        return {
            "score": 0.0,
            "feedback": f"Query failed to execute: {error}",
            "query_executes": False,
            "output_matches": False,
            "breakdown": {"execute": 0.0, "partial_rows": 0.0, "exact": 0.0, "efficiency": 0.0},
        }

    execute_score = 0.30

    expected_norm = _normalize_rows(EXPECTED_ROWS)
    actual_norm   = _normalize_rows(rows)

    matched_rows   = sum(1 for r in actual_norm if r in expected_norm)
    total_expected = len(expected_norm)
    partial_score  = 0.40 * (matched_rows / total_expected) if total_expected > 0 else 0.0
    exact_score    = 0.30 if actual_norm == expected_norm else 0.0

    wasted_steps      = max(0, step_count - 1)
    efficiency_penalty = min(0.15, wasted_steps * 0.05)

    raw_score   = execute_score + partial_score + exact_score - efficiency_penalty
    final_score = round(max(0.01, min(0.99, raw_score)), 4)

    if exact_score > 0:
        feedback = "Perfect! Query output matches exactly."
    elif partial_score > 0:
        feedback = f"Query runs but output is partially correct: {matched_rows}/{total_expected} rows matched."
    else:
        feedback = "Query runs but output does not match expected results."

    if efficiency_penalty > 0:
        feedback += f" Efficiency penalty applied: -{efficiency_penalty:.2f}."

    return {
        "score": final_score,
        "feedback": feedback,
        "query_executes": True,
        "output_matches": exact_score > 0,
        "breakdown": {
            "execute":      execute_score,
            "partial_rows": round(partial_score, 4),
            "exact":        exact_score,
            "efficiency":   -round(efficiency_penalty, 4),
        },
    }
