# 🧠 Example Agent Trajectory

This document demonstrates exactly what a successful reinforcement learning episode looks like when a baseline AI agent solves **Task 3 (Hard)** in the SQL Repair Environment.

Notice how the environment provides dense rewards (partial points) giving the agent feedback that it is making the correct schema mutations *before* it even writes a SQL query.

---

## 🏁 Episode Initialization

The agent calls `POST /reset?task_id=task3_hard`.

The environment spins up a fresh, isolated SQLite database in memory and returns the observation:

```json
{
  "task_id": "task3_hard",
  "goal": "A multi-table schema exists: 'customers', 'products', and 'transactions'. However, the data is extremely messy. You must drop nulls from all 3 tables, remove duplicate rows in transactions, and fix the typo 'quanity' to 'qty'. Finally, calculate total revenue per region for revenue > 10000 by joining all tables.",
  "max_steps": 15,
  "tables": [
    { "table_name": "transactions", "row_count": 35, "columns": ["transaction_id", "customer_id", "product_id", "quanity"] },
    { "table_name": "customers", "row_count": 12, "columns": ["cust_id", "cust_name", "region"] },
    { "table_name": "products", "row_count": 8, "columns": ["prod_id", "prod_name", "price"] }
  ]
}
```

---

## 🛠️ Step 1: Fixing the Typo

The agent uses its "Data Cleaner" logic to fix the `quanity` column typo.

**Action (Agent Payload):**
```json
{
  "action_type": "rename_column",
  "table_name": "transactions",
  "column_name": "quanity",
  "new_column_name": "qty"
}
```

**Environment Response:**
```json
{
  "reward": {
    "step_reward": 0.15,
    "total_reward": 0.15,
    "efficiency_penalty": 0.0,
    "query_executes": false
  },
  "observation": {
    "last_action": "RENAME_COLUMN",
    "step_count": 1
  }
}
```

---

## 🧹 Steps 2 - 5: Data Cleaning

The agent dynamically issues 4 separate actions to clean up the bad schema records and drop duplicates.

1.  `DROP_NULLS` on `customers.cust_name` 👉 *(Reward +0.10)*
2.  `DROP_NULLS` on `products.prod_name` 👉 *(Reward +0.10)*
3.  `DROP_DUPLICATES` on `transactions` 👉 *(Reward +0.10)*
4.  `DROP_NULLS` on `transactions.customer_id` 👉 *(Reward +0.10)*

**Total Accumulated Reward:** `0.55 / 1.0`

---

## 🚀 Step 6: Query Submission

Now that the data schema is perfectly sanitized across all three tables, the agent transitions to its "SQL Writer" phase and submits the complex mult-table `JOIN` query.

**Action (Agent Payload):**
```json
{
  "action_type": "submit_query",
  "query": "SELECT c.region, SUM(p.price * t.qty) AS total_revenue FROM transactions t JOIN customers c ON t.customer_id = c.cust_id JOIN products p ON t.product_id = p.prod_id GROUP BY c.region HAVING total_revenue > 10000 ORDER BY total_revenue DESC"
}
```

**Environment Response:**
```json
{
  "reward": {
    "step_reward": 0.45,
    "total_reward": 1.0000,
    "efficiency_penalty": 0.0,
    "query_executes": true,
    "schema_correct": true,
    "output_matches": true
  },
  "observation": {
    "done": true,
    "current_score": 1.0,
    "last_query_result": [
      {"region": "North America", "total_revenue": 14500.0},
      {"region": "Europe", "total_revenue": 11200.0}
    ]
  }
}
```

### 🏆 Outcome

The episode successfully terminates (`done: true`). The agent has achieved a perfect **1.0 total score** in exactly 6 steps by systematically resolving data quality issues before computing the complex analytics.
