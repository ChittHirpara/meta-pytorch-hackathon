"""
Task 1 — EASY
Scenario: A single table with one broken SQL query (syntax error / wrong column name).
Agent only needs to submit one corrected query.
"""

import sqlite3
from typing import Any, Dict, Tuple


TASK_ID = "task1_easy"
TASK_NAME = "Fix SQL Syntax Error"
DIFFICULTY = "easy"
MAX_STEPS = 5
DESCRIPTION = (
    "A table called 'employees' exists with columns: id, name, department, salary. "
    "The provided SQL query has a syntax/typo error. Fix it and submit the correct query."
)

INSTRUCTIONS = (
    "Available actions:\n"
    "1. submit_query — Submit your corrected SQL query string.\n"
    "2. done — Signal that you are finished.\n\n"
    "Hint: Look carefully at the SQL keywords and column names in the broken query."
)

# Ground truth data that will be loaded into the DB
SEED_DATA = [
    (1, "Alice",   "Engineering", 95000),
    (2, "Bob",     "Marketing",   72000),
    (3, "Carol",   "Engineering", 88000),
    (4, "Dave",    "HR",          61000),
    (5, "Eve",     "Marketing",   75000),
    (6, "Frank",   "Engineering", 91000),
    (7, "Grace",   "HR",          63000),
    (8, "Heidi",   "Marketing",   69000),
]

# The broken query the agent must fix
# Bug: "SELCT" instead of "SELECT", and "deprtment" instead of "department"
BROKEN_QUERY = (
    "SELCT id, name, deprtment, salary "
    "FROM employees "
    "WHERE salary > 70000 "
    "ORDER BY salary DESC"
)

# The correct query (used by grader as ground truth)
CORRECT_QUERY = (
    "SELECT id, name, department, salary "
    "FROM employees "
    "WHERE salary > 70000 "
    "ORDER BY salary DESC"
)

EXPECTED_ROWS = [
    {"id": 1, "name": "Alice",   "department": "Engineering", "salary": 95000},
    {"id": 6, "name": "Frank",   "department": "Engineering", "salary": 91000},
    {"id": 3, "name": "Carol",   "department": "Engineering", "salary": 88000},
    {"id": 5, "name": "Eve",     "department": "Marketing",   "salary": 75000},
    {"id": 2, "name": "Bob",     "department": "Marketing",   "salary": 72000},
]


def setup_database(conn: sqlite3.Connection) -> None:
    """Create and populate the employees table."""
    conn.execute("DROP TABLE IF EXISTS employees")
    conn.execute("""
        CREATE TABLE employees (
            id         INTEGER PRIMARY KEY,
            name       TEXT    NOT NULL,
            department TEXT    NOT NULL,
            salary     INTEGER NOT NULL
        )
    """)
    conn.executemany(
        "INSERT INTO employees (id, name, department, salary) VALUES (?, ?, ?, ?)",
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
        "tables": ["employees"],
        "action_types_used": ["submit_query", "done"],
    }


def run_query(conn: sqlite3.Connection, query: str) -> Tuple[bool, list, str]:
    """Execute a query and return (success, rows, error)."""
    try:
        cursor = conn.execute(query)
        cols = [d[0] for d in cursor.description]
        rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
        return True, rows, ""
    except Exception as e:
        return False, [], str(e)
