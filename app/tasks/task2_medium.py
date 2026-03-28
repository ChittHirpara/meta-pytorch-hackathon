"""
Task 2 — MEDIUM
Scenario: A table called 'orders' has dirty data (nulls, duplicates, wrong types)
AND a SQL query with a logical bug (wrong WHERE clause + wrong column name in GROUP BY).
Agent must clean data first, then fix and submit the query.
"""

import sqlite3
from typing import Any, Dict, Tuple


TASK_ID = "task2_medium"
TASK_NAME = "Clean Data and Fix Query Logic"
DIFFICULTY = "medium"
MAX_STEPS = 10
DESCRIPTION = (
    "A table called 'orders' has: null values in 'customer_name', "
    "duplicate rows, a column named 'amt' that should be 'amount', "
    "and some non-numeric values in 'amount'. "
    "Additionally, the provided SQL query has a logical error in the WHERE clause. "
    "Clean the data, then fix and submit the correct query."
)

INSTRUCTIONS = (
    "Available actions:\n"
    "1. drop_nulls      — Drop rows where a column is null. Provide 'column_name'.\n"
    "2. drop_duplicates — Remove duplicate rows from the table.\n"
    "3. rename_column   — Rename a column. Provide 'column_name' and 'new_column_name'.\n"
    "4. cast_column     — Cast column to a type. Provide 'column_name' and 'cast_to'.\n"
    "5. submit_query    — Submit your fixed SQL query.\n"
    "6. done            — Signal that you are finished.\n\n"
    "Hint: Fix the data issues first, then look at the query logic carefully. "
    "The goal is total revenue per department for orders above $100."
)

SEED_DATA = [
    (1,  "Alice",   "Electronics", "250",   "completed"),
    (2,  "Bob",     "Clothing",    "85",    "completed"),
    (3,  None,      "Electronics", "310",   "completed"),
    (4,  "Dave",    "Clothing",    "120",   "completed"),
    (5,  "Eve",     "Electronics", "400",   "completed"),
    (6,  "Frank",   "Furniture",   "N/A",   "pending"),
    (7,  "Grace",   "Clothing",    "95",    "completed"),
    (8,  "Heidi",   "Electronics", "275",   "completed"),
    (4,  "Dave",    "Clothing",    "120",   "completed"),
    (5,  "Eve",     "Electronics", "400",   "completed"),
    (9,  None,      "Furniture",   "500",   "completed"),
    (10, "Ivan",    "Furniture",   "150",   "completed"),
]

BROKEN_QUERY = (
    "SELECT department, SUM(amt) AS total "
    "FROM orders "
    "WHERE amt > 50 AND status = 'completed' "
    "GROUP BY department "
    "ORDER BY total DESC"
)

CORRECT_QUERY = (
    "SELECT department, SUM(amount) AS total_revenue "
    "FROM orders "
    "WHERE amount > 100 AND status = 'completed' "
    "GROUP BY department "
    "ORDER BY total_revenue DESC"
)

EXPECTED_ROWS = [
    {"department": "Electronics", "total_revenue": 1235.0},
    {"department": "Furniture",   "total_revenue": 650.0},
    {"department": "Clothing",    "total_revenue": 240.0},
]


def setup_database(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS orders")
    conn.execute("""
        CREATE TABLE orders (
            id            INTEGER,
            customer_name TEXT,
            department    TEXT,
            amt           TEXT,
            status        TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO orders (id, customer_name, department, amt, status) VALUES (?,?,?,?,?)",
        SEED_DATA,
    )
    conn.commit()


def get_task_info() -> Dict[str, Any]:
    return {
        "task_id":    TASK_ID,
        "name":       TASK_NAME,
        "difficulty": DIFFICULTY,
        "max_steps":  MAX_STEPS,
        "description": DESCRIPTION,
        "instructions": INSTRUCTIONS,
        "broken_query": BROKEN_QUERY,
        "tables": ["orders"],
        "action_types_used": [
            "drop_nulls", "drop_duplicates",
            "rename_column", "cast_column",
            "submit_query", "done"
        ],
    }


def run_query(conn: sqlite3.Connection, query: str) -> Tuple[bool, list, str]:
    try:
        cursor = conn.execute(query)
        cols = [d[0] for d in cursor.description]
        rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
        return True, rows, ""
    except Exception as e:
        return False, [], str(e)
