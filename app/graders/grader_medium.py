"""
Grader — Task 2 (Medium)
Scores both data cleaning actions AND the final query.
Returns a score between 0.0 and 1.0.
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
    """
    Check how well the agent cleaned the data.
    Returns sub-scores for each cleaning task.
    """
    scores = {}

    # 1. Was 'amt' renamed to 'amount'?
    try:
        conn.execute("SELECT amount FROM orders LIMIT 1")
        scores["rename_column"] = 0.15
    except Exception:
        scores["rename_column"] = 0.0

    # 2. Were null customer_names dropped?
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM orders WHERE customer_name IS NULL")
        null_count = cursor.fetchone()[0]
        scores["drop_nulls"] = 0.15 if null_count == 0 else 0.0
    except Exception:
        scores["drop_nulls"] = 0.0

    # 3. Were duplicates removed?
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM orders")
        total = cursor.fetchone()[0]
        # After dedup + null drop, should have <= 8 rows
        scores["drop_duplicates"] = 0.10 if total <= 8 else 0.0
    except Exception:
        scores["drop_duplicates"] = 0.0

    # 4. Was amount cast to numeric AND bad rows removed?
    # Fix: only award credit if:
    #   (a) the 'amount' column exists (was renamed from 'amt')
    #   (b) no non-numeric values remain (the 'N/A' row was removed/cast)
    try:
        # Check that no non-numeric string values remain in amount column
        cursor = conn.execute(
            """SELECT COUNT(*) FROM orders
               WHERE CAST(amount AS REAL) IS NULL
               AND amount IS NOT NULL"""
        )
        bad_count = cursor.fetchone()[0]
        # Also verify at least one numeric row exists (column was actually cast)
        cursor2 = conn.execute(
            "SELECT COUNT(*) FROM orders WHERE CAST(amount AS REAL) IS NOT NULL"
        )
        numeric_count = cursor2.fetchone()[0]
        # Award credit only if no bad rows remain AND numeric rows exist
        scores["cast_column"] = 0.10 if (bad_count == 0 and numeric_count > 0) else 0.0
    except Exception:
        scores["cast_column"] = 0.0

    return scores


def grade(
    conn: sqlite3.Connection,
    submitted_query: str,
    step_count: int,
    max_steps: int,
    cleaning_actions_taken: List[str],
) -> Dict[str, Any]:
    """
    Grade Task 2 submission.

    Scoring breakdown:
      +0.15  renamed 'amt' → 'amount'
      +0.15  dropped null customer_names
      +0.10  dropped duplicate rows
      +0.10  handled bad 'N/A' amount value
      +0.15  query executes without error
      +0.20  partial row match on query output
      +0.15  exact match on query output
      -0.05  per wasted step beyond step 4
    """

    # ── Data quality scores ──────────────────────────────────────────────────
    dq = grade_data_quality(conn)
    data_score = sum(dq.values())

    # ── Query execution ──────────────────────────────────────────────────────
    success, rows, error = run_query(conn, submitted_query)
    if not success:
        efficiency_penalty = min(0.20, max(0, step_count - 4) * 0.05)
        final_score = round(max(0.01, min(0.99, data_score - efficiency_penalty)), 4)
        return {
            "score": final_score,
            "feedback": f"Data cleaning partial credit: {data_score:.2f}. Query failed: {error}",
            "query_executes": False,
            "output_matches": False,
            "data_quality": dq,
            "breakdown": {"data_quality": data_score, "execute": 0.0, "partial_rows": 0.0, "exact": 0.0},
        }

    execute_score = 0.15

    # ── Row matching ─────────────────────────────────────────────────────────
    expected_norm = _normalize_rows(EXPECTED_ROWS)
    actual_norm   = _normalize_rows(rows)

    matched = sum(1 for r in actual_norm if r in expected_norm)
    total   = len(expected_norm)
    partial_score = 0.20 * (matched / total) if total > 0 else 0.0
    exact_score   = 0.15 if actual_norm == expected_norm else 0.0

    # ── Efficiency penalty ───────────────────────────────────────────────────
    wasted = max(0, step_count - 4)
    penalty = min(0.20, wasted * 0.05)

    raw = data_score + execute_score + partial_score + exact_score - penalty
    final_score = round(max(0.01, min(0.99, raw)), 4)

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
