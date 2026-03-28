"""
Task 3 — HARD
Scenario: Three tables (customers, products, transactions) with schema errors,
dirty data, and a broken multi-table JOIN + aggregation query.
"""

import sqlite3
from typing import Any, Dict, Tuple


TASK_ID = "task3_hard"
TASK_NAME = "Multi-Table Join Repair and Aggregation Fix"
DIFFICULTY = "hard"
MAX_STEPS = 15
DESCRIPTION = (
    "Three tables exist: 'customers' (cust_id, cust_name, region), "
    "'products' (prod_id, prod_name, category, price), "
    "and 'transactions' (txn_id, customer_id, product_id, qty, txn_date). "
    "Problems: 'transactions' has a column 'quanity' (typo) instead of 'qty', "
    "null values across tables, duplicate transactions, "
    "and the provided query uses AVG instead of SUM, wrong HAVING threshold, "
    "wrong JOIN key, and wrong ORDER BY. "
    "Clean all tables and submit a corrected query that returns total revenue "
    "per region for regions with revenue > 10000, ordered by revenue descending."
)

INSTRUCTIONS = (
    "Available actions:\n"
    "1. drop_nulls      — Drop rows where a column is null. Provide 'table_name' and 'column_name'.\n"
    "2. drop_duplicates — Remove duplicate rows. Provide 'table_name'.\n"
    "3. rename_column   — Rename a column. Provide 'table_name', 'column_name', 'new_column_name'.\n"
    "4. cast_column     — Cast column to type. Provide 'table_name', 'column_name', 'cast_to'.\n"
    "5. submit_query    — Submit your corrected SQL query.\n"
    "6. done            — Signal finished.\n\n"
    "Hint: Fix 'quanity' -> 'qty' first. Then clean nulls/dupes in all 3 tables. "
    "The correct query joins all 3 tables, computes SUM(price * qty) per region, "
    "filters HAVING total_revenue > 10000, orders DESC."
)

CUSTOMERS_DATA = [
    (1,  "Alice",   "North"),
    (2,  "Bob",     "South"),
    (3,  "Carol",   "East"),
    (4,  "Dave",    "West"),
    (5,  "Eve",     "North"),
    (6,  None,      "South"),
    (7,  "Grace",   "East"),
    (8,  "Heidi",   "West"),
    (9,  "Ivan",    "North"),
    (10, "Judy",    "South"),
]

PRODUCTS_DATA = [
    (1,  "Laptop",   "Electronics", 1200.0),
    (2,  "Phone",    "Electronics",  800.0),
    (3,  "Desk",     "Furniture",    350.0),
    (4,  "Chair",    "Furniture",    150.0),
    (5,  "Shirt",    "Clothing",      45.0),
    (6,  "Pants",    "Clothing",      60.0),
    (7,  None,       "Electronics",  500.0),
    (8,  "Monitor",  "Electronics",  400.0),
]

TRANSACTIONS_DATA = [
    (1,  1,  1, 2,    "2024-01-10"),
    (2,  2,  3, 4,    "2024-01-11"),
    (3,  3,  2, 1,    "2024-01-12"),
    (4,  4,  4, 10,   "2024-01-13"),
    (5,  5,  1, 3,    "2024-01-14"),
    (6,  6,  5, 5,    "2024-01-15"),
    (7,  7,  8, 2,    "2024-01-16"),
    (8,  8,  3, 6,    "2024-01-17"),
    (9,  9,  2, 2,    "2024-01-18"),
    (10, 10, 6, 3,    "2024-01-19"),
    (5,  5,  1, 3,    "2024-01-14"),
    (3,  3,  2, 1,    "2024-01-12"),
    (11, 1,  8, 5,    "2024-01-20"),
    (12, 2,  1, 1,    "2024-01-21"),
    (13, None, 2, 1,  "2024-01-22"),
]

BROKEN_QUERY = (
    "SELECT c.region,\n"
    "       AVG(p.price * t.quanity) AS total_revenue\n"
    "FROM transactions t\n"
    "JOIN customers c ON t.cust_id = c.cust_id\n"
    "JOIN products  p ON t.product_id = p.prod_id\n"
    "WHERE t.quanity IS NOT NULL\n"
    "GROUP BY c.region\n"
    "HAVING total_revenue > 500\n"
    "ORDER BY total_revenue ASC"
)

CORRECT_QUERY = (
    "SELECT c.region,\n"
    "       SUM(p.price * t.qty) AS total_revenue\n"
    "FROM transactions t\n"
    "JOIN customers c ON t.customer_id = c.cust_id\n"
    "JOIN products  p ON t.product_id = p.prod_id\n"
    "GROUP BY c.region\n"
    "HAVING total_revenue > 10000\n"
    "ORDER BY total_revenue DESC"
)

EXPECTED_ROWS = [
    {"region": "North", "total_revenue": 15600.0},
]


def setup_database(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS customers")
    conn.execute("""
        CREATE TABLE customers (
            cust_id   INTEGER PRIMARY KEY,
            cust_name TEXT,
            region    TEXT NOT NULL
        )
    """)
    conn.executemany(
        "INSERT INTO customers (cust_id, cust_name, region) VALUES (?,?,?)",
        CUSTOMERS_DATA,
    )

    conn.execute("DROP TABLE IF EXISTS products")
    conn.execute("""
        CREATE TABLE products (
            prod_id   INTEGER PRIMARY KEY,
            prod_name TEXT,
            category  TEXT NOT NULL,
            price     REAL NOT NULL
        )
    """)
    conn.executemany(
        "INSERT INTO products (prod_id, prod_name, category, price) VALUES (?,?,?,?)",
        PRODUCTS_DATA,
    )

    conn.execute("DROP TABLE IF EXISTS transactions")
    conn.execute("""
        CREATE TABLE transactions (
            txn_id      INTEGER,
            customer_id INTEGER,
            product_id  INTEGER,
            quanity     INTEGER,
            txn_date    TEXT
        )
    """)
    extra = [
        (20, 1, 1, 5, "2024-02-01"),
    ]
    conn.executemany(
        "INSERT INTO transactions (txn_id, customer_id, product_id, quanity, txn_date) VALUES (?,?,?,?,?)",
        TRANSACTIONS_DATA + extra,
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
        "tables": ["customers", "products", "transactions"],
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
