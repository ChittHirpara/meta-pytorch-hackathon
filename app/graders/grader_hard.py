"""
Grader — Task 3 (Hard)
Scores schema fixes, multi-table data cleaning, and the final complex query.
"""

import sqlite3
from typing import Any, Dict, List

from app.tasks.task3_hard import EXPECTED_ROWS, run_query


def _normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for row in rows:
        norm = {}
        for k, v in row.items():
            if isinstance(v, str):
                norm[k] = v.strip().lower()
            elif isinstance(v, float):
                norm[k] = round(v, 1)
            else:
                norm[k] = v
        normalized.append(norm)
    return normalized


def _normalize_expected() -> List[Dict[str, Any]]:
    return [
        {k: (v.strip().lower() if isinstance(v, str) else (round(v, 1) if isinstance(v, float) else v))
         for k, v in row.items()}
        for row in EXPECTED_ROWS
    ]


def grade_schema(conn: sqlite3.Connection) -> Dict[str, float]:
    scores = {}

    try:
        conn.execute("SELECT qty FROM transactions LIMIT 1")
        scores["rename_qty"] = 0.15
    except Exception:
        scores["rename_qty"] = 0.0

    try:
        cur = conn.execute("SELECT COUNT(*) FROM customers WHERE cust_name IS NULL")
        scores["drop_null_customers"] = 0.10 if cur.fetchone()[0] == 0 else 0.0
    except Exception:
        scores["drop_null_customers"] = 0.0

    try:
        cur = conn.execute("SELECT COUNT(*) FROM products WHERE prod_name IS NULL")
        scores["drop_null_products"] = 0.05 if cur.fetchone()[0] == 0 else 0.0
    except Exception:
        scores["drop_null_products"] = 0.0

    try:
        cur = conn.execute("""
            SELECT COUNT(*) FROM (
                SELECT txn_id, customer_id, product_id, COUNT(*) c
                FROM transactions
                GROUP BY txn_id, customer_id, product_id
                HAVING c > 1
            )
        """)
        scores["drop_dup_transactions"] = 0.10 if cur.fetchone()[0] == 0 else 0.0
    except Exception:
        scores["drop_dup_transactions"] = 0.05

    try:
        cur = conn.execute("SELECT COUNT(*) FROM transactions WHERE customer_id IS NULL")
        scores["drop_null_txn"] = 0.05 if cur.fetchone()[0] == 0 else 0.0
    except Exception:
        scores["drop_null_txn"] = 0.0

    return scores


def grade(
    conn: sqlite3.Connection,
    submitted_query: str,
    step_count: int,
    max_steps: int,
    cleaning_actions_taken: List[str],
) -> Dict[str, Any]:
    schema      = grade_schema(conn)
    schema_score = sum(schema.values())

    success, rows, error = run_query(conn, submitted_query)
    if not success:
        penalty = min(0.20, max(0, step_count - 6) * 0.05)
        final   = round(max(0.01, min(0.99, schema_score - penalty)), 4)
        return {
            "score": final,
            "feedback": f"Schema/cleaning score: {schema_score:.2f}. Query failed: {error}",
            "query_executes": False,
            "output_matches": False,
            "schema_scores": schema,
            "breakdown": {"schema": schema_score, "execute": 0.0, "partial": 0.0, "exact": 0.0},
        }

    execute_score = 0.15

    expected_norm = _normalize_expected()
    actual_norm   = _normalize_rows(rows)

    matched = sum(1 for r in actual_norm if r in expected_norm)
    total   = len(expected_norm)
    partial = 0.20 * (matched / total) if total > 0 else 0.0
    exact   = 0.20 if actual_norm == expected_norm else 0.0

    wasted  = max(0, step_count - 6)
    penalty = min(0.20, wasted * 0.05)

    raw   = schema_score + execute_score + partial + exact - penalty
    final = round(max(0.01, min(0.99, raw)), 4)

    if exact > 0:
        feedback = "Outstanding! All tables cleaned and complex query is perfect."
    elif partial > 0:
        feedback = f"Query runs, {matched}/{total} regions matched. Schema score: {schema_score:.2f}."
    else:
        feedback = f"Query runs but output doesn't match. Schema score: {schema_score:.2f}."

    if penalty > 0:
        feedback += f" Efficiency penalty: -{penalty:.2f}."

    return {
        "score": final,
        "feedback": feedback,
        "query_executes": True,
        "output_matches": exact > 0,
        "schema_scores": schema,
        "breakdown": {
            "schema":    round(schema_score, 4),
            "execute":   execute_score,
            "partial":   round(partial, 4),
            "exact":     exact,
            "efficiency": -round(penalty, 4),
        },
    }
