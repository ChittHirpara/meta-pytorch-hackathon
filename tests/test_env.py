"""
tests/test_env.py — Comprehensive unit + integration tests for the SQL Repair Environment.

Tests cover:
  - Environment reset and observation structure for all 3 tasks
  - Every action type (submit_query, drop_nulls, drop_duplicates, rename_column, cast_column, done)
  - Grader logic for easy / medium / hard tasks
  - Reward correctness (no dampening, no hardcoded zeros)
  - Edge cases (bad query, missing fields, double-done)
  - FastAPI endpoints via TestClient with lifespan (uses context manager)
"""

import sqlite3
import pytest

from fastapi.testclient import TestClient

from app.main import app
from app.environment import SQLRepairEnvironment
from app.models import Action, ActionType
from app.graders import grader_easy, grader_medium, grader_hard
from app.tasks import task1_easy, task2_medium, task3_hard

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    """Fresh environment, not yet reset."""
    return SQLRepairEnvironment()


@pytest.fixture
def env_easy(env):
    env.reset(task_id="task1_easy")
    return env


@pytest.fixture
def env_medium(env):
    env.reset(task_id="task2_medium")
    return env


@pytest.fixture
def env_hard(env):
    env.reset(task_id="task3_hard")
    return env


@pytest.fixture
def client():
    """
    TestClient used as a context manager so FastAPI lifespan runs,
    which initialises the global _env before any endpoint is called.
    """
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# 1. Health / API endpoint tests
# ---------------------------------------------------------------------------

class TestAPIEndpoints:

    def test_health_check(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_info_endpoint(self, client):
        resp = client.get("/info")
        assert resp.status_code == 200
        assert len(resp.json()["tasks"]) == 3

    def test_tasks_endpoint(self, client):
        resp = client.get("/tasks")
        assert resp.status_code == 200
        ids = [t["task_id"] for t in resp.json()["tasks"]]
        assert "task1_easy"   in ids
        assert "task2_medium" in ids
        assert "task3_hard"   in ids

    def test_reset_default(self, client):
        resp = client.post("/reset")
        assert resp.status_code == 200
        obs = resp.json()
        assert obs["task_id"] == "task1_easy"
        assert obs["done"] is False

    def test_reset_all_tasks(self, client):
        for task_id in ["task1_easy", "task2_medium", "task3_hard"]:
            resp = client.post("/reset", params={"task_id": task_id})
            assert resp.status_code == 200, f"Reset failed for {task_id}: {resp.text}"
            assert resp.json()["task_id"] == task_id

    def test_reset_invalid_task(self, client):
        resp = client.post("/reset", params={"task_id": "task99_unknown"})
        assert resp.status_code == 400

    def test_state_after_reset(self, client):
        client.post("/reset", params={"task_id": "task1_easy"})
        resp = client.get("/state")
        assert resp.status_code == 200
        state = resp.json()
        assert state["task_id"]    == "task1_easy"
        assert state["step_count"] == 0

    def test_validate_endpoint(self, client):
        resp = client.get("/validate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert "task1_easy" in data["tasks"]

    def test_step_done_action(self, client):
        client.post("/reset", params={"task_id": "task1_easy"})
        resp = client.post("/step", json={"action_type": "done"})
        assert resp.status_code == 200

    def test_step_submit_correct_query_easy(self, client):
        client.post("/reset", params={"task_id": "task1_easy"})
        payload = {
            "action_type": "submit_query",
            "query": (
                "SELECT id, name, department, salary FROM employees "
                "WHERE salary > 70000 ORDER BY salary DESC"
            )
        }
        resp = client.post("/step", json=payload)
        assert resp.status_code == 200
        result = resp.json()
        assert result["reward"]["total_reward"] >= 0.5
        assert result["reward"]["output_matches"] is True
        assert result["done"] is True

    def test_state_endpoint_reflects_steps(self, client):
        client.post("/reset", params={"task_id": "task2_medium"})
        client.post("/step", json={
            "action_type": "drop_nulls",
            "column_name": "customer_name",
            "table_name":  "orders"
        })
        resp  = client.get("/state")
        state = resp.json()
        assert state["step_count"]         == 1
        assert len(state["action_history"]) == 1


# ---------------------------------------------------------------------------
# 2. Environment reset tests
# ---------------------------------------------------------------------------

class TestEnvironmentReset:

    def test_reset_returns_observation(self, env):
        obs = env.reset(task_id="task1_easy")
        assert obs.task_id           == "task1_easy"
        assert obs.difficulty.value  == "easy"
        assert obs.done              is False
        assert obs.step_count        == 0
        assert obs.max_steps         == 5
        assert len(obs.tables)       == 1
        assert obs.tables[0].table_name == "employees"

    def test_reset_clears_state(self, env_easy):
        env_easy.step(Action(action_type=ActionType.DONE))
        obs = env_easy.reset(task_id="task1_easy")
        assert obs.step_count    == 0
        assert obs.current_score == 0.0
        assert obs.done          is False

    def test_reset_medium_has_correct_tables(self, env):
        obs = env.reset(task_id="task2_medium")
        tables = {t.table_name for t in obs.tables}
        assert "orders" in tables
        assert obs.max_steps    == 10

    def test_reset_hard_has_three_tables(self, env):
        obs    = env.reset(task_id="task3_hard")
        tables = {t.table_name for t in obs.tables}
        assert tables       == {"customers", "products", "transactions"}
        assert obs.max_steps == 15

    def test_reset_populates_broken_query(self, env_easy):
        obs = env_easy._build_observation()
        assert "SELCT" in obs.broken_query or "deprtment" in obs.broken_query


# ---------------------------------------------------------------------------
# 3. Action handler tests (unit-level, no HTTP)
# ---------------------------------------------------------------------------

class TestActionHandlers:

    # ── submit_query ─────────────────────────────────────────────────────────

    def test_submit_correct_query_easy(self, env_easy):
        action = Action(
            action_type=ActionType.SUBMIT_QUERY,
            query=(
                "SELECT id, name, department, salary FROM employees "
                "WHERE salary > 70000 ORDER BY salary DESC"
            )
        )
        result = env_easy.step(action)
        assert result.reward.total_reward  >= 0.5
        assert result.reward.output_matches is True
        assert result.done                  is True

    def test_submit_broken_query_gives_negative_reward(self, env_easy):
        action = Action(
            action_type=ActionType.SUBMIT_QUERY,
            query="SELCT * FORM employees"
        )
        result = env_easy.step(action)
        assert result.reward.step_reward < 0
        assert result.done               is False

    def test_submit_query_no_query_string(self, env_easy):
        result = env_easy.step(Action(action_type=ActionType.SUBMIT_QUERY))
        assert result.reward.step_reward < 0

    # ── drop_nulls ───────────────────────────────────────────────────────────

    def test_drop_nulls_medium(self, env_medium):
        # Verify nulls exist before
        obs_before    = env_medium._build_observation()
        orders_before = next(t for t in obs_before.tables if t.table_name == "orders")
        cn_null_before = next(
            c.null_count for c in orders_before.columns if c.name == "customer_name"
        )
        assert cn_null_before > 0

        result = env_medium.step(Action(
            action_type=ActionType.DROP_NULLS,
            column_name="customer_name",
            table_name="orders"
        ))
        assert result.reward.step_reward == pytest.approx(0.10, abs=0.01)

        # Verify nulls gone
        orders_after  = next(t for t in result.observation.tables if t.table_name == "orders")
        cn_null_after = next(c.null_count for c in orders_after.columns if c.name == "customer_name")
        assert cn_null_after == 0

    def test_drop_nulls_missing_column_name(self, env_medium):
        result = env_medium.step(Action(
            action_type=ActionType.DROP_NULLS,
            table_name="orders"
        ))
        assert result.reward.step_reward < 0

    # ── drop_duplicates ──────────────────────────────────────────────────────

    def test_drop_duplicates_medium(self, env_medium):
        result = env_medium.step(Action(
            action_type=ActionType.DROP_DUPLICATES,
            table_name="orders"
        ))
        assert result.reward.step_reward == pytest.approx(0.10, abs=0.01)

    def test_drop_duplicates_hard_transactions(self, env_hard):
        result = env_hard.step(Action(
            action_type=ActionType.DROP_DUPLICATES,
            table_name="transactions"
        ))
        assert result.reward.step_reward == pytest.approx(0.10, abs=0.01)

    # ── rename_column ────────────────────────────────────────────────────────

    def test_rename_column_medium(self, env_medium):
        result = env_medium.step(Action(
            action_type=ActionType.RENAME_COLUMN,
            column_name="amt",
            new_column_name="amount",
            table_name="orders"
        ))
        assert result.reward.step_reward == pytest.approx(0.15, abs=0.01)

    def test_rename_column_hard_quantity_typo(self, env_hard):
        """
        transactions.quanity (typo) → transactions.qty
        table_name must be passed so the correct table is targeted.
        """
        result = env_hard.step(Action(
            action_type=ActionType.RENAME_COLUMN,
            column_name="quanity",
            new_column_name="qty",
            table_name="transactions"
        ))
        assert result.reward.step_reward == pytest.approx(0.15, abs=0.01)

    def test_rename_column_missing_names(self, env_medium):
        result = env_medium.step(Action(
            action_type=ActionType.RENAME_COLUMN,
            table_name="orders"
        ))
        assert result.reward.step_reward < 0

    # ── cast_column ──────────────────────────────────────────────────────────

    def test_cast_column_after_rename(self, env_medium):
        env_medium.step(Action(
            action_type=ActionType.RENAME_COLUMN,
            column_name="amt",
            new_column_name="amount",
            table_name="orders"
        ))
        result = env_medium.step(Action(
            action_type=ActionType.CAST_COLUMN,
            column_name="amount",
            cast_to="REAL",
            table_name="orders"
        ))
        assert result.reward.step_reward == pytest.approx(0.10, abs=0.01)

    # ── done ─────────────────────────────────────────────────────────────────

    def test_done_with_no_progress_gives_penalty(self, env_easy):
        result = env_easy.step(Action(action_type=ActionType.DONE))
        assert result.reward.step_reward < 0
        assert result.done is True

    def test_step_after_done_raises(self, env_easy):
        env_easy.step(Action(action_type=ActionType.DONE))
        with pytest.raises(RuntimeError):
            env_easy.step(Action(action_type=ActionType.DONE))


# ---------------------------------------------------------------------------
# 4. Reward correctness tests (the critical bug fixes)
# ---------------------------------------------------------------------------

class TestRewardCorrectness:

    def test_no_reward_dampening_easy(self, env_easy):
        """
        Fix 1: total_reward must NOT be halved by *0.5.
        A perfect submit_query should give total_reward >= 0.9, not ~0.5.
        """
        result = env_easy.step(Action(
            action_type=ActionType.SUBMIT_QUERY,
            query=(
                "SELECT id, name, department, salary FROM employees "
                "WHERE salary > 70000 ORDER BY salary DESC"
            )
        ))
        assert result.reward.total_reward >= 0.9, (
            f"Reward dampening bug still present! total_reward={result.reward.total_reward}"
        )

    def test_data_quality_score_populated(self, env_medium):
        """
        Fix 2: data_quality_score must not always be 0.0 after cleaning + grading.
        """
        env_medium.step(Action(action_type=ActionType.DROP_NULLS,
                               column_name="customer_name", table_name="orders"))
        env_medium.step(Action(action_type=ActionType.DROP_DUPLICATES, table_name="orders"))
        env_medium.step(Action(action_type=ActionType.RENAME_COLUMN,
                               column_name="amt", new_column_name="amount",
                               table_name="orders"))
        env_medium.step(Action(action_type=ActionType.CAST_COLUMN,
                               column_name="amount", cast_to="REAL",
                               table_name="orders"))
        result = env_medium.step(Action(
            action_type=ActionType.SUBMIT_QUERY,
            query=(
                "SELECT department, SUM(amount) AS total_revenue FROM orders "
                "WHERE amount > 100 AND status = 'completed' "
                "GROUP BY department ORDER BY total_revenue DESC"
            )
        ))
        assert result.reward.data_quality_score > 0.0, (
            f"data_quality_score is still 0.0! "
            f"Got: {result.reward.data_quality_score}"
        )

    def test_output_matches_flag(self, env_easy):
        """
        Fix 3: output_matches must come from grader, not hardcoded.
        """
        result = env_easy.step(Action(
            action_type=ActionType.SUBMIT_QUERY,
            query=(
                "SELECT id, name, department, salary FROM employees "
                "WHERE salary > 70000 ORDER BY salary DESC"
            )
        ))
        assert result.reward.output_matches is True
        assert result.reward.task_complete  is True

    def test_cleaning_rewards_accumulate_correctly(self, env_medium):
        """
        After 2 cleaning steps (+0.10 each), total_reward should be ~0.20.
        The old *0.5 bug would give ~0.10 instead.
        """
        env_medium.step(Action(action_type=ActionType.DROP_NULLS,
                               column_name="customer_name", table_name="orders"))
        r2 = env_medium.step(Action(action_type=ActionType.DROP_DUPLICATES,
                                    table_name="orders"))
        assert r2.reward.total_reward >= 0.18, (
            f"Reward accumulation wrong: total={r2.reward.total_reward} after 2 cleaning steps"
        )


# ---------------------------------------------------------------------------
# 5. Grader unit tests
# ---------------------------------------------------------------------------

class TestGraderEasy:

    def setup_method(self):
        self.conn = sqlite3.connect(":memory:")
        task1_easy.setup_database(self.conn)

    def teardown_method(self):
        self.conn.close()

    def test_correct_query_scores_high(self):
        result = grader_easy.grade(
            self.conn,
            "SELECT id, name, department, salary FROM employees "
            "WHERE salary > 70000 ORDER BY salary DESC",
            step_count=1,
            max_steps=5,
        )
        assert result["score"]          >= 0.9
        assert result["output_matches"] is True
        assert result["query_executes"] is True

    def test_wrong_query_scores_differently(self):
        result = grader_easy.grade(
            self.conn,
            "SELECT * FROM employees",
            step_count=1,
            max_steps=5,
        )
        assert result["output_matches"] is False

    def test_syntax_error_scores_zero(self):
        result = grader_easy.grade(
            self.conn,
            "SELCT * FORM employees",
            step_count=1,
            max_steps=5,
        )
        assert result["score"]          == 0.0
        assert result["query_executes"] is False

    def test_efficiency_penalty_applied(self):
        q = ("SELECT id, name, department, salary FROM employees "
             "WHERE salary > 70000 ORDER BY salary DESC")
        r1 = grader_easy.grade(self.conn, q, step_count=1, max_steps=5)
        r4 = grader_easy.grade(self.conn, q, step_count=4, max_steps=5)
        assert r1["score"] > r4["score"]


class TestGraderMedium:

    def setup_method(self):
        self.conn = sqlite3.connect(":memory:")
        task2_medium.setup_database(self.conn)
        # Apply full cleaning sequence
        self.conn.execute("DELETE FROM orders WHERE customer_name IS NULL")
        self.conn.execute("""
            DELETE FROM orders WHERE rowid NOT IN (
                SELECT MIN(rowid) FROM orders
                GROUP BY id, customer_name, department, amt, status
            )
        """)
        self.conn.execute("ALTER TABLE orders RENAME COLUMN amt TO amount")
        self.conn.execute(
            "UPDATE orders SET amount = CAST(amount AS REAL) "
            "WHERE CAST(amount AS REAL) IS NOT NULL"
        )
        self.conn.execute(
            "DELETE FROM orders "
            "WHERE CAST(amount AS REAL) IS NULL AND amount IS NOT NULL"
        )
        self.conn.commit()

    def teardown_method(self):
        self.conn.close()

    def test_correct_query_scores_high(self):
        result = grader_medium.grade(
            self.conn,
            "SELECT department, SUM(amount) AS total_revenue FROM orders "
            "WHERE amount > 100 AND status = 'completed' "
            "GROUP BY department ORDER BY total_revenue DESC",
            step_count=4,
            max_steps=10,
            cleaning_actions_taken=["drop_nulls", "drop_duplicates", "rename", "cast"],
        )
        assert result["score"] >= 0.7

    def test_cleaning_data_quality_detected(self):
        scores = grader_medium.grade_data_quality(self.conn)
        assert scores.get("rename_column",    0) > 0
        assert scores.get("drop_nulls",       0) > 0
        assert scores.get("drop_duplicates",  0) > 0


class TestGraderHard:

    def setup_method(self):
        self.conn = sqlite3.connect(":memory:")
        task3_hard.setup_database(self.conn)
        # Apply full cleaning sequence
        self.conn.execute("ALTER TABLE transactions RENAME COLUMN quanity TO qty")
        self.conn.execute("DELETE FROM customers    WHERE cust_name    IS NULL")
        self.conn.execute("DELETE FROM products     WHERE prod_name    IS NULL")
        self.conn.execute("""
            DELETE FROM transactions WHERE rowid NOT IN (
                SELECT MIN(rowid) FROM transactions
                GROUP BY txn_id, customer_id, product_id
            )
        """)
        self.conn.execute("DELETE FROM transactions WHERE customer_id IS NULL")
        self.conn.commit()

    def teardown_method(self):
        self.conn.close()

    def test_correct_query_scores_high(self):
        result = grader_hard.grade(
            self.conn,
            "SELECT c.region, SUM(p.price * t.qty) AS total_revenue "
            "FROM transactions t "
            "JOIN customers c ON t.customer_id = c.cust_id "
            "JOIN products  p ON t.product_id  = p.prod_id "
            "GROUP BY c.region HAVING total_revenue > 10000 "
            "ORDER BY total_revenue DESC",
            step_count=6,
            max_steps=15,
            cleaning_actions_taken=["rename", "drop_nulls", "drop_duplicates"],
        )
        assert result["score"]          >= 0.8
        assert result["output_matches"] is True

    def test_schema_grader_detects_rename(self):
        scores = grader_hard.grade_schema(self.conn)
        assert scores.get("rename_qty", 0) > 0

    def test_schema_grader_detects_null_cleanup(self):
        scores = grader_hard.grade_schema(self.conn)
        assert scores.get("drop_null_customers", 0) > 0


# ---------------------------------------------------------------------------
# 6. Edge case / robustness tests
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_unknown_action_type_via_http(self, client):
        client.post("/reset", params={"task_id": "task1_easy"})
        resp = client.post("/step", json={"action_type": "explode_database"})
        assert resp.status_code == 422  # FastAPI validation error

    def test_episode_id_is_unique(self, env):
        obs1 = env.reset("task1_easy")
        obs2 = env.reset("task1_easy")
        assert obs1.episode_id != obs2.episode_id

    def test_max_steps_reached_ends_episode(self, env_easy):
        """Easy has max_steps=5 — exhaust via done actions."""
        result = None
        for _ in range(5):
            result = env_easy.step(Action(action_type=ActionType.DONE))
            if result.done:
                break
        assert result.done is True

    def test_observation_contains_null_counts(self, env_medium):
        obs    = env_medium._build_observation()
        orders = next(t for t in obs.tables if t.table_name == "orders")
        cn_col = next(c for c in orders.columns if c.name == "customer_name")
        assert cn_col.null_count == 2   # task2 has 2 null customer_name rows

    def test_full_episode_easy_perfect_score(self, env_easy):
        """One-shot correct query → perfect score."""
        result = env_easy.step(Action(
            action_type=ActionType.SUBMIT_QUERY,
            query=(
                "SELECT id, name, department, salary FROM employees "
                "WHERE salary > 70000 ORDER BY salary DESC"
            )
        ))
        assert result.reward.total_reward  >= 0.9
        assert result.reward.output_matches is True
        assert result.done                  is True

    def test_full_episode_medium_high_score(self, env_medium):
        """Full cleaning sequence + correct query → high score."""
        env_medium.step(Action(action_type=ActionType.DROP_NULLS,
                               column_name="customer_name", table_name="orders"))
        env_medium.step(Action(action_type=ActionType.DROP_DUPLICATES,
                               table_name="orders"))
        env_medium.step(Action(action_type=ActionType.RENAME_COLUMN,
                               column_name="amt", new_column_name="amount",
                               table_name="orders"))
        env_medium.step(Action(action_type=ActionType.CAST_COLUMN,
                               column_name="amount", cast_to="REAL",
                               table_name="orders"))
        result = env_medium.step(Action(
            action_type=ActionType.SUBMIT_QUERY,
            query=(
                "SELECT department, SUM(amount) AS total_revenue FROM orders "
                "WHERE amount > 100 AND status = 'completed' "
                "GROUP BY department ORDER BY total_revenue DESC"
            )
        ))
        assert result.reward.total_reward  >= 0.7, (
            f"Expected >= 0.7, got {result.reward.total_reward}"
        )
        assert result.reward.output_matches is True

    def test_full_episode_hard_high_score(self, env_hard):
        """Full hard task cleaning + correct JOIN query → high score."""
        env_hard.step(Action(action_type=ActionType.RENAME_COLUMN,
                             column_name="quanity", new_column_name="qty",
                             table_name="transactions"))
        env_hard.step(Action(action_type=ActionType.DROP_NULLS,
                             column_name="cust_name", table_name="customers"))
        env_hard.step(Action(action_type=ActionType.DROP_NULLS,
                             column_name="prod_name", table_name="products"))
        env_hard.step(Action(action_type=ActionType.DROP_DUPLICATES,
                             table_name="transactions"))
        env_hard.step(Action(action_type=ActionType.DROP_NULLS,
                             column_name="customer_id", table_name="transactions"))
        result = env_hard.step(Action(
            action_type=ActionType.SUBMIT_QUERY,
            query=(
                "SELECT c.region, SUM(p.price * t.qty) AS total_revenue "
                "FROM transactions t "
                "JOIN customers c ON t.customer_id = c.cust_id "
                "JOIN products  p ON t.product_id  = p.prod_id "
                "GROUP BY c.region HAVING total_revenue > 10000 "
                "ORDER BY total_revenue DESC"
            )
        ))
        assert result.reward.total_reward  >= 0.7, (
            f"Expected >= 0.7, got {result.reward.total_reward}"
        )
        assert result.reward.output_matches is True
