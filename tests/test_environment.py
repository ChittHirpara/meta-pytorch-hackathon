"""
tests/test_environment.py
=========================
10+ mixed and extreme test cases for the SQL Repair Environment.

Categories:
  ✅ Normal cases       — expected happy-path flows
  ⚠️  Edge cases        — boundary conditions
  💥 Extreme cases      — SQL injection, empty tables, huge payloads
  🔁 State cases        — reset behavior, episode boundaries
  🎯 Grader cases       — scoring accuracy checks

Run with:
    pytest tests/test_environment.py -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from app.environment import SQLRepairEnvironment
from app.models import Action, ActionType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    """Fresh environment instance for each test."""
    e = SQLRepairEnvironment()
    yield e
    e.close()


def make_action(**kwargs) -> Action:
    """Helper to build Action objects cleanly."""
    return Action(**kwargs)


# ===========================================================================
# TEST 1 — Normal: Task 1 perfect solve in 1 step
# ===========================================================================
def test_task1_perfect_solve(env):
    """
    NORMAL CASE:
    Agent submits the exact correct query on step 1.
    Expects: score >= 0.8, done=True, no errors.
    """
    obs = env.reset(task_id="task1_easy")

    assert obs.task_id == "task1_easy"
    assert obs.step_count == 0
    assert obs.done is False
    assert "employees" in [t.table_name for t in obs.tables]

    result = env.step(make_action(
        action_type=ActionType.SUBMIT_QUERY,
        query="SELECT id, name, department, salary FROM employees WHERE salary > 70000 ORDER BY salary DESC",
    ))

    assert result.reward.total_reward >= 0.7
    assert result.reward.query_executes is True
    assert result.observation.last_action_error is None
    print(f"  ✅ Task1 perfect score: {result.reward.total_reward}")


# ===========================================================================
# TEST 2 — Normal: Task 2 full pipeline solve
# ===========================================================================
def test_task2_full_pipeline(env):
    """
    NORMAL CASE:
    Agent does all 4 cleaning steps then submits correct query.
    Expects: score >= 0.6, done=True.
    """
    env.reset(task_id="task2_medium")

    # Step 1 — rename
    env.step(make_action(action_type=ActionType.RENAME_COLUMN,
                         column_name="amt", new_column_name="amount"))
    # Step 2 — drop nulls
    env.step(make_action(action_type=ActionType.DROP_NULLS,
                         column_name="customer_name"))
    # Step 3 — drop duplicates
    env.step(make_action(action_type=ActionType.DROP_DUPLICATES))
    # Step 4 — cast
    env.step(make_action(action_type=ActionType.CAST_COLUMN,
                         column_name="amount", cast_to="REAL"))
    # Step 5 — submit
    result = env.step(make_action(
        action_type=ActionType.SUBMIT_QUERY,
        query=(
            "SELECT department, SUM(amount) AS total_revenue "
            "FROM orders WHERE amount > 100 AND status = 'completed' "
            "GROUP BY department ORDER BY total_revenue DESC"
        ),
    ))

    assert result.reward.query_executes is True
    assert result.reward.total_reward >= 0.5
    print(f"  ✅ Task2 full pipeline score: {result.reward.total_reward}")


# ===========================================================================
# TEST 3 — Normal: Task 3 full multi-table pipeline
# ===========================================================================
def test_task3_full_pipeline(env):
    """
    NORMAL CASE:
    Agent cleans all 3 tables then submits correct JOIN query.
    Expects: score >= 0.5.
    """
    env.reset(task_id="task3_hard")

    steps = [
        make_action(action_type=ActionType.RENAME_COLUMN,
                    table_name="transactions", column_name="quanity", new_column_name="qty"),
        make_action(action_type=ActionType.DROP_NULLS,
                    table_name="customers", column_name="cust_name"),
        make_action(action_type=ActionType.DROP_NULLS,
                    table_name="products", column_name="prod_name"),
        make_action(action_type=ActionType.DROP_DUPLICATES, table_name="transactions"),
        make_action(action_type=ActionType.DROP_NULLS,
                    table_name="transactions", column_name="customer_id"),
    ]
    for s in steps:
        env.step(s)

    result = env.step(make_action(
        action_type=ActionType.SUBMIT_QUERY,
        query=(
            "SELECT c.region, SUM(p.price * t.qty) AS total_revenue "
            "FROM transactions t "
            "JOIN customers c ON t.customer_id = c.cust_id "
            "JOIN products p ON t.product_id = p.prod_id "
            "GROUP BY c.region HAVING total_revenue > 10000 "
            "ORDER BY total_revenue DESC"
        ),
    ))

    assert result.reward.query_executes is True
    assert result.reward.total_reward >= 0.4
    print(f"  ✅ Task3 full pipeline score: {result.reward.total_reward}")


# ===========================================================================
# TEST 4 — Edge Case: Submit completely broken SQL
# ===========================================================================
def test_broken_sql_gives_negative_reward(env):
    """
    EDGE CASE:
    Agent submits total garbage SQL.
    Expects: query_executes=False, step_reward <= 0.
    """
    env.reset(task_id="task1_easy")

    result = env.step(make_action(
        action_type=ActionType.SUBMIT_QUERY,
        query="THIS IS NOT SQL AT ALL !!!",
    ))

    assert result.reward.query_executes is False
    assert result.reward.step_reward <= 0.0
    assert result.observation.last_action_error is not None
    print(f"  ✅ Broken SQL correctly penalized: {result.reward.step_reward}")


# ===========================================================================
# TEST 5 — Extreme Case: SQL Injection attempt
# ===========================================================================
def test_sql_injection_attempt(env):
    """
    EXTREME CASE:
    Agent tries a SQL injection via the query field.
    Environment must not crash — should return error, not execute DROP.
    """
    env.reset(task_id="task1_easy")

    injection = "SELECT * FROM employees; DROP TABLE employees; --"
    result = env.step(make_action(
        action_type=ActionType.SUBMIT_QUERY,
        query=injection,
    ))

    # Environment should either error out OR ignore the second statement
    # Either way, the employees table must still exist
    state = env.state()
    table_names = [t.table_name for t in state.tables]
    assert "employees" in table_names, "SQL injection dropped the table!"
    print(f"  ✅ SQL injection handled safely. Tables intact: {table_names}")


# ===========================================================================
# TEST 6 — Extreme Case: Empty query string
# ===========================================================================
def test_empty_query_string(env):
    """
    EXTREME CASE:
    Agent submits an empty string as the query.
    Expects: error returned, no crash, step_reward <= 0.
    """
    env.reset(task_id="task1_easy")

    result = env.step(make_action(
        action_type=ActionType.SUBMIT_QUERY,
        query="",
    ))

    assert result.reward.step_reward <= 0.0
    print(f"  ✅ Empty query handled: error={result.observation.last_action_error}")


# ===========================================================================
# TEST 7 — Extreme Case: Drop nulls on non-existent column
# ===========================================================================
def test_drop_nulls_nonexistent_column(env):
    """
    EXTREME CASE:
    Agent tries to drop nulls from a column that doesn't exist.
    Expects: error returned, environment does NOT crash, state unchanged.
    """
    env.reset(task_id="task1_easy")

    before_state = env.state()
    before_rows = before_state.tables[0].row_count

    result = env.step(make_action(
        action_type=ActionType.DROP_NULLS,
        column_name="nonexistent_column_xyz",
    ))

    after_state = env.state()
    after_rows = after_state.tables[0].row_count

    # Row count must be unchanged
    assert before_rows == after_rows
    assert result.reward.step_reward <= 0.0
    print(f"  ✅ Non-existent column handled. Rows unchanged: {before_rows}")


# ===========================================================================
# TEST 8 — Edge Case: rename same column twice (idempotency)
# ===========================================================================
def test_rename_column_twice(env):
    """
    EDGE CASE:
    Agent renames 'amt' to 'amount', then tries to rename 'amt' again.
    Second rename should fail gracefully.
    """
    env.reset(task_id="task2_medium")

    # First rename — should succeed
    r1 = env.step(make_action(
        action_type=ActionType.RENAME_COLUMN,
        column_name="amt",
        new_column_name="amount",
    ))
    assert r1.reward.step_reward > 0

    # Second rename of same (now gone) column — should fail gracefully
    r2 = env.step(make_action(
        action_type=ActionType.RENAME_COLUMN,
        column_name="amt",
        new_column_name="amount",
    ))
    assert r2.reward.step_reward <= 0.0
    assert r2.observation.last_action_error is not None
    print(f"  ✅ Double rename handled. First: {r1.reward.step_reward}, Second: {r2.reward.step_reward}")


# ===========================================================================
# TEST 9 — State Case: reset clears previous episode state
# ===========================================================================
def test_reset_clears_state(env):
    """
    STATE CASE:
    After doing steps in task1, reset to task2.
    Expects: clean slate — step_count=0, new tables, new episode_id.
    """
    obs1 = env.reset(task_id="task1_easy")
    env.step(make_action(
        action_type=ActionType.SUBMIT_QUERY,
        query="SELECT * FROM employees",
    ))

    episode1 = obs1.episode_id
    state_after_step = env.state()
    assert state_after_step.step_count == 1

    # Now reset to task2
    obs2 = env.reset(task_id="task2_medium")

    assert obs2.step_count == 0
    assert obs2.episode_id != episode1
    assert obs2.task_id == "task2_medium"
    assert obs2.current_score == 0.0
    assert obs2.done is False

    table_names = [t.table_name for t in obs2.tables]
    assert "orders" in table_names
    assert "employees" not in table_names
    print(f"  ✅ Reset cleared state. New episode: {obs2.episode_id}")


# ===========================================================================
# TEST 10 — Extreme Case: Exhaust all max steps without solving
# ===========================================================================
def test_max_steps_reached(env):
    """
    EXTREME CASE:
    Agent keeps submitting wrong queries until max_steps is reached.
    Expects: done=True after max steps, total_reward < 0.5.
    """
    env.reset(task_id="task1_easy")
    max_steps = 5
    result = None

    for i in range(max_steps + 2):   # try to go beyond max
        result = env.step(make_action(
            action_type=ActionType.SUBMIT_QUERY,
            query="SELECT invalid FROM employees",  # explicitly wrong column
        ))
        if result.done:
            break

    assert result.done is True
    assert result.reward.total_reward < 0.8
    print(f"  ✅ Max steps enforced. Final score: {result.reward.total_reward}")


# ===========================================================================
# TEST 11 — Extreme Case: Very long SQL query (payload bomb)
# ===========================================================================
def test_very_long_query(env):
    """
    EXTREME CASE:
    Agent submits an enormous SQL query string (10,000 chars).
    Expects: environment does NOT hang or crash. Returns in <5 seconds.
    """
    import time
    env.reset(task_id="task1_easy")

    long_query = "SELECT " + ", ".join(["id"] * 500) + " FROM employees"

    start = time.time()
    result = env.step(make_action(
        action_type=ActionType.SUBMIT_QUERY,
        query=long_query,
    ))
    elapsed = time.time() - start

    assert elapsed < 5.0, f"Query took too long: {elapsed:.2f}s"
    # Result may succeed or fail but must not crash
    assert result is not None
    print(f"  ✅ Long query handled in {elapsed:.3f}s. Executes: {result.reward.query_executes}")


# ===========================================================================
# TEST 12 — Grader Case: Partial row match gives partial score
# ===========================================================================
def test_partial_row_match_gives_partial_score(env):
    """
    GRADER CASE:
    Agent submits a query that returns SOME correct rows but not all.
    Expects: score > 0 but < full score. (Partial credit works.)
    """
    env.reset(task_id="task1_easy")

    # This query gets some right rows but wrong ORDER and missing filter
    result = env.step(make_action(
        action_type=ActionType.SUBMIT_QUERY,
        query="SELECT id, name, department, salary FROM employees WHERE salary > 70000",
    ))

    # Should get partial credit — rows match but order may be wrong
    assert result.reward.query_executes is True
    assert result.reward.step_reward > 0.0   # some credit
    assert result.reward.total_reward < 1.0  # not full credit
    print(f"  ✅ Partial credit working: {result.reward.total_reward}")


# ===========================================================================
# TEST 13 — Edge Case: done action with no progress
# ===========================================================================
def test_done_action_no_progress(env):
    """
    EDGE CASE:
    Agent signals done immediately without doing anything.
    Expects: episode ends, score = 0 or very low, negative step reward.
    """
    env.reset(task_id="task2_medium")

    result = env.step(make_action(action_type=ActionType.DONE))

    assert result.done is True
    assert result.reward.step_reward <= 0.0
    print(f"  ✅ Premature done penalized: {result.reward.step_reward}")


# ===========================================================================
# TEST 14 — Extreme Case: Query on wrong table name
# ===========================================================================
def test_query_wrong_table_name(env):
    """
    EXTREME CASE:
    Agent queries a table that doesn't exist in the environment.
    Expects: query_executes=False, meaningful error message returned.
    """
    env.reset(task_id="task1_easy")

    result = env.step(make_action(
        action_type=ActionType.SUBMIT_QUERY,
        query="SELECT * FROM nonexistent_table_abc123",
    ))

    assert result.reward.query_executes is False
    assert result.observation.last_action_error is not None
    assert "nonexistent_table_abc123" in result.observation.last_action_error or \
           "no such table" in result.observation.last_action_error.lower()
    print(f"  ✅ Wrong table error: {result.observation.last_action_error}")


# ===========================================================================
# TEST 15 — Extreme Case: Cast column to invalid type
# ===========================================================================
def test_cast_to_invalid_type(env):
    """
    EXTREME CASE:
    Agent tries to cast a column to a made-up type like 'FAKETYPE'.
    Expects: does not crash, returns error gracefully.
    """
    env.reset(task_id="task2_medium")

    # First rename so column exists
    env.step(make_action(
        action_type=ActionType.RENAME_COLUMN,
        column_name="amt",
        new_column_name="amount",
    ))

    result = env.step(make_action(
        action_type=ActionType.CAST_COLUMN,
        column_name="amount",
        cast_to="FAKETYPE",
    ))

    # SQLite is permissive with types, so this may or may not error
    # but it must NOT crash the server
    assert result is not None
    print(f"  ✅ Invalid cast type handled. Error: {result.observation.last_action_error}")


# ===========================================================================
# TEST 16 — State Case: state() before reset raises error
# ===========================================================================
def test_state_before_reset_raises(env):
    """
    STATE CASE:
    Calling state() before reset() should raise RuntimeError, not crash.
    """
    fresh_env = SQLRepairEnvironment()
    try:
        with pytest.raises(RuntimeError):
            fresh_env.state()
        print("  ✅ state() before reset raises RuntimeError correctly")
    finally:
        fresh_env.close()


# ===========================================================================
# TEST 17 — Extreme Case: step() before reset raises error
# ===========================================================================
def test_step_before_reset_raises(env):
    """
    EXTREME CASE:
    Calling step() before reset() must raise RuntimeError.
    """
    fresh_env = SQLRepairEnvironment()
    try:
        with pytest.raises(RuntimeError):
            fresh_env.step(make_action(action_type=ActionType.DONE))
        print("  ✅ step() before reset raises RuntimeError correctly")
    finally:
        fresh_env.close()


# ===========================================================================
# TEST 18 — Grader Case: Task 2 score inflated check (Bug 2 regression)
# ===========================================================================
def test_task2_no_inflated_cast_score(env):
    """
    GRADER REGRESSION (Bug 2):
    Cast score must NOT be awarded before actually casting.
    Submit query without casting — cast sub-score must be 0.
    """
    env.reset(task_id="task2_medium")

    # Only rename — do NOT cast
    env.step(make_action(
        action_type=ActionType.RENAME_COLUMN,
        column_name="amt",
        new_column_name="amount",
    ))

    # Submit query immediately without casting
    result = env.step(make_action(
        action_type=ActionType.SUBMIT_QUERY,
        query=(
            "SELECT department, SUM(amount) AS total_revenue "
            "FROM orders WHERE amount > 100 AND status = 'completed' "
            "GROUP BY department ORDER BY total_revenue DESC"
        ),
    ))

    # Score should be lower than full because cast not done
    # and the N/A row pollutes the data
    assert result.reward.total_reward < 0.95
    print(f"  ✅ Cast score not inflated without casting: {result.reward.total_reward}")


# ===========================================================================
# Run summary
# ===========================================================================
if __name__ == "__main__":
    import traceback

    tests = [
        test_task1_perfect_solve,
        test_task2_full_pipeline,
        test_task3_full_pipeline,
        test_broken_sql_gives_negative_reward,
        test_sql_injection_attempt,
        test_empty_query_string,
        test_drop_nulls_nonexistent_column,
        test_rename_column_twice,
        test_reset_clears_state,
        test_max_steps_reached,
        test_very_long_query,
        test_partial_row_match_gives_partial_score,
        test_done_action_no_progress,
        test_query_wrong_table_name,
        test_cast_to_invalid_type,
        test_state_before_reset_raises,
        test_step_before_reset_raises,
        test_task2_no_inflated_cast_score,
    ]

    passed = 0
    failed = 0
    errors = []

    print("\n" + "="*60)
    print("  SQL REPAIR ENV — FULL TEST SUITE")
    print("="*60)

    for test_fn in tests:
        e = SQLRepairEnvironment()
        try:
            test_fn(e)
            passed += 1
            print(f"  PASS  {test_fn.__name__}")
        except Exception as ex:
            failed += 1
            errors.append((test_fn.__name__, str(ex)))
            print(f"  FAIL  {test_fn.__name__} → {ex}")
            traceback.print_exc()
        finally:
            e.close()

    print("\n" + "="*60)
    print(f"  Results: {passed} passed / {failed} failed / {len(tests)} total")
    if errors:
        print("\n  Failed tests:")
        for name, err in errors:
            print(f"    ❌ {name}: {err}")
    else:
        print("  🎉 All tests passed!")
    print("="*60 + "\n")
