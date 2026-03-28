"""
Grader — Task 2 (Medium)
Scores both data cleaning actions AND the final query.
"""

import sqlite3
from typing import Any, Dict, List

from app.tasks.task2_medium import EXPECTED_ROWS, run_query


def _normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for row in rows:
        norm = {}
        for k, v in row.items():
            if isinstance(v, str):
                norm[k] = v.strip().lower()
            elif isinstance(v, float):
                norm[k] = round(v, 2)
            else:
                norm[k] = v
        normalized.append(norm)
    return normalized


def grade_data_quality(conn: sqlite3.Connection) -> Dict[str, float]:
    scores = {}

    try:
        conn.execute("SELECT amount FROM orders LIMIT 1")
        scores["rename_column"] = 0.15
    except Exception:
        scores["rename_column"] = 0.0

    try:
        cursor = conn.execute("SELECT COUNT(*) FROM orders WHERE customer_name IS NULL")
        scores["drop_nulls"] = 0.15 if cursor.fetchone()[0] == 0 else 0.0
    except Exception:
        scores["drop_nulls"] = 0.0

    try:
        cursor = conn.execute("SELECT COUNT(*) FROM orders")
        total = cursor.fetchone()[0]
        scores["drop_duplicates"] = 0.10 if total <= 8 else 0.0
    except Exception:
        scores["drop_duplicates"] = 0.0

    try:
        conn.execute("SELECT amount FROM orders WHERE amount NOT GLOB '*[^0-9.]*' OR amount IS NULL LIMIT 1")
        scores["cast_column"] = 0.10
    except Exception:
        scores["cast_column"] = 0.05

    return scores


def grade(
    conn: sqlite3.Connection,
    submitted_query: str,
    step_count: int,
    max_steps: int,
    cleaning_actions_taken: List[str],
) -> Dict[str, Any]:
    dq = grade_data_quality(conn)
    data_score = sum(dq.values())

    success, rows, error = run_query(conn, submitted_query)
    if not success:
        efficiency_penalty = min(0.20, max(0, step_count - 4) * 0.05)
        final_score = round(max(0.0, data_score - efficiency_penalty), 4)
        return {
            "score": final_score,
            "feedback": f"Data cleaning partial credit: {data_score:.2f}. Query failed: {error}",
            "query_executes": False,
            "output_matches": False,
            "data_quality": dq,
            "breakdown": {"data_quality": data_score, "execute": 0.0, "partial_rows": 0.0, "exact": 0.0},
        }

    execute_score = 0.15

    expected_norm = _normalize_rows(EXPECTED_ROWS)
    actual_norm   = _normalize_rows(rows)

    matched = sum(1 for r in actual_norm if r in expected_norm)
    total   = len(expected_norm)
    partial_score = 0.20 * (matched / total) if total > 0 else 0.0
    exact_score   = 0.15 if actual_norm == expected_norm else 0.0

    wasted  = max(0, step_count - 4)
    penalty = min(0.20, wasted * 0.05)

    raw         = data_score + execute_score + partial_score + exact_score - penalty
    final_score = round(max(0.0, min(1.0, raw)), 4)

    if exact_score > 0:
        feedback = "Excellent! Data cleaned and query output matches perfectly."
    elif partial_score > 0:
        feedback = f"Query runs, {matched}/{total} output rows match. Data quality: {data_score:.2f}."
    else:
        feedback = f"Query runs but output doesn't match. Data quality score: {data_score:.2f}."

    if penalty > 0:
        feedback += f" Efficiency penalty: -{penalty:.2f}."

    return {
        "score": final_score,
        "feedback": feedback,
        "query_executes": True,
        "output_matches": exact_score > 0,
        "data_quality": dq,
        "breakdown": {
            "data_quality":  round(data_score, 4),
            "execute":       execute_score,
            "partial_rows":  round(partial_score, 4),
            "exact":         exact_score,
            "efficiency":    -round(penalty, 4),
        },
    }
