"""
environment.py — Core OpenEnv SQL Repair Environment

Manages episode state, applies actions, computes rewards, and
exposes reset() / step() / state() as the standard OpenEnv interface.
"""

from __future__ import annotations

import sqlite3
import uuid
from typing import Any, Dict, List, Optional

from app.models import (
    Action, ActionType, ColumnInfo, EnvironmentState,
    Observation, QueryResult, Reward, StepResult,
    TableInfo, TaskDifficulty,
)
from app.tasks import task1_easy, task2_medium, task3_hard
from app.graders import grader_easy, grader_medium, grader_hard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TASK_MODULES = {
    "task1_easy":   task1_easy,
    "task2_medium": task2_medium,
    "task3_hard":   task3_hard,
}

GRADER_MODULES = {
    "task1_easy":   grader_easy,
    "task2_medium": grader_medium,
    "task3_hard":   grader_hard,
}

DIFFICULTY_MAP = {
    "task1_easy":   TaskDifficulty.EASY,
    "task2_medium": TaskDifficulty.MEDIUM,
    "task3_hard":   TaskDifficulty.HARD,
}


def _get_table_info(conn: sqlite3.Connection, table_name: str) -> TableInfo:
    """Inspect a SQLite table and return a TableInfo snapshot."""
    # Row count
    row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

    # Column info
    pragma = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    columns: List[ColumnInfo] = []
    for col in pragma:
        col_name = col[1]
        col_type = col[2] or "TEXT"
        null_count = conn.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE {col_name} IS NULL"
        ).fetchone()[0]
        sample_cur = conn.execute(
            f"SELECT {col_name} FROM {table_name} WHERE {col_name} IS NOT NULL LIMIT 3"
        )
        samples = [r[0] for r in sample_cur.fetchall()]
        columns.append(ColumnInfo(
            name=col_name,
            dtype=col_type,
            nullable=null_count > 0,
            null_count=null_count,
            sample_values=samples,
        ))

    # Preview rows
    preview_cur = conn.execute(f"SELECT * FROM {table_name} LIMIT 5")
    col_names = [d[0] for d in preview_cur.description]
    preview_rows = [dict(zip(col_names, row)) for row in preview_cur.fetchall()]

    return TableInfo(
        table_name=table_name,
        row_count=row_count,
        columns=columns,
        preview_rows=preview_rows,
    )


def _run_query_safe(conn: sqlite3.Connection, query: str) -> QueryResult:
    """Run a SQL query and return a QueryResult (never raises)."""
    try:
        cur = conn.execute(query)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        return QueryResult(success=True, rows=rows, row_count=len(rows), columns=cols)
    except Exception as exc:
        return QueryResult(success=False, error_message=str(exc))


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class SQLRepairEnvironment:
    """
    OpenEnv-compliant SQL Repair Environment.

    Usage:
        env = SQLRepairEnvironment()
        obs = env.reset(task_id="task1_easy")
        result = env.step(action)
        state  = env.state()
    """

    def __init__(self) -> None:
        self._conn: Optional[sqlite3.Connection] = None
        self._task_id: Optional[str] = None
        self._task_module = None
        self._episode_id: str = ""
        self._step_count: int = 0
        self._max_steps: int = 10
        self._total_reward: float = 0.0
        self._done: bool = False
        self._action_history: List[str] = []
        self._cleaning_actions: List[str] = []
        self._last_query_result: Optional[QueryResult] = None
        self._last_error: Optional[str] = None
        self._last_action: Optional[str] = None
        # Populated by grader on submit_query — used to build Reward correctly
        self._last_grader_result: dict = {}

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(self, task_id: str = "task1_easy") -> Observation:
        """Start a fresh episode for the given task."""
        if task_id not in TASK_MODULES:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASK_MODULES)}")

        # Fresh in-memory SQLite for this episode
        if self._conn:
            self._conn.close()
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)

        self._task_id      = task_id
        self._task_module  = TASK_MODULES[task_id]
        self._episode_id   = str(uuid.uuid4())[:8]
        self._step_count   = 0
        self._max_steps    = self._task_module.MAX_STEPS
        self._total_reward = 0.0
        self._done         = False
        self._action_history  = []
        self._cleaning_actions = []
        self._last_query_result = None
        self._last_error   = None
        self._last_action  = None
        self._last_grader_result = {}

        # Populate the database
        self._task_module.setup_database(self._conn)

        return self._build_observation()

    # ── step ─────────────────────────────────────────────────────────────────

    def step(self, action: Action) -> StepResult:
        """Apply an action and return the next observation + reward."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._conn is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self._step_count += 1
        self._last_error  = None
        self._last_action = action.action_type.value
        self._last_query_result = None

        # ── Dispatch action ──────────────────────────────────────────────────
        step_reward = 0.0
        feedback    = ""

        if action.action_type == ActionType.DONE:
            step_reward, feedback = self._handle_done(action)

        elif action.action_type == ActionType.SUBMIT_QUERY:
            step_reward, feedback = self._handle_submit_query(action)

        elif action.action_type == ActionType.DROP_NULLS:
            step_reward, feedback = self._handle_drop_nulls(action)

        elif action.action_type == ActionType.DROP_DUPLICATES:
            step_reward, feedback = self._handle_drop_duplicates(action)

        elif action.action_type == ActionType.RENAME_COLUMN:
            step_reward, feedback = self._handle_rename_column(action)

        elif action.action_type == ActionType.CAST_COLUMN:
            step_reward, feedback = self._handle_cast_column(action)

        elif action.action_type == ActionType.CLEAN_COLUMN:
            step_reward, feedback = self._handle_clean_column(action)

        else:
            self._last_error = f"Unknown action type: {action.action_type}"
            step_reward = -0.05

        # ── Check max steps ──────────────────────────────────────────────────
        if self._step_count >= self._max_steps and not self._done:
            self._done  = True
            step_reward -= 0.10   # penalty for not finishing in time
            feedback    += " [Max steps reached — episode ended]"

        # ── Accumulate reward ────────────────────────────────────────────────
        # FIX: removed the erroneous *0.5 dampening — step_reward is already 0‑1
        self._total_reward = round(min(0.999, max(0.001, self._total_reward + step_reward)), 4)
        self._action_history.append(
            f"Step {self._step_count}: {action.action_type.value} → {feedback[:80]}"
        )

        # ── Build return objects ─────────────────────────────────────────────
        gr = self._last_grader_result  # populated by _handle_submit_query
        obs    = self._build_observation(feedback=feedback)
        reward = Reward(
            step_reward        = round(step_reward, 4),
            total_reward       = self._total_reward,
            # FIX: use grader breakdown for correctness — fallback to row-count heuristic
            correctness_score  = round(gr.get("breakdown", {}).get("exact", 0.0)
                                       + gr.get("breakdown", {}).get("partial_rows",
                                           gr.get("breakdown", {}).get("partial", 0.0)), 4)
                                 if gr else (
                                     min(1.0, self._last_query_result.row_count / 10)
                                     if self._last_query_result and self._last_query_result.success
                                     else 0.0
                                 ),
            # FIX: pull data_quality_score from grader result instead of hardcoding 0.0
            data_quality_score = round(
                sum(gr.get("data_quality", gr.get("schema_scores", {})).values())
                if gr else 0.0, 4
            ),
            efficiency_penalty = max(0.0, (self._step_count / self._max_steps) - 0.5),
            feedback           = feedback,
            query_executes     = self._last_query_result.success if self._last_query_result else False,
            # FIX: schema_correct derived from grader schema score instead of always False
            schema_correct     = bool(gr.get("schema_scores", {}).get("rename_qty", 0.0) > 0)
                                 if gr else False,
            output_matches     = bool(gr.get("output_matches", False)) if gr else False,
            task_complete      = bool(gr.get("output_matches", False)) if gr else False,
        )

        return StepResult(
            observation = obs,
            reward      = reward,
            done        = self._done,
            info        = {"step": self._step_count, "feedback": feedback},
        )

    # ── state ────────────────────────────────────────────────────────────────

    def state(self) -> EnvironmentState:
        """Return the current internal state snapshot."""
        if self._conn is None or self._task_id is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        tables = self._get_all_tables()
        return EnvironmentState(
            episode_id     = self._episode_id,
            task_id        = self._task_id,
            difficulty     = DIFFICULTY_MAP[self._task_id],
            step_count     = self._step_count,
            max_steps      = self._max_steps,
            total_reward   = self._total_reward,
            done           = self._done,
            tables         = tables,
            broken_query   = self._task_module.BROKEN_QUERY,
            action_history = self._action_history,
        )

    # ── Action handlers ──────────────────────────────────────────────────────

    def _handle_submit_query(self, action: Action):
        if not action.query:
            self._last_error = "No query provided for submit_query action."
            self._last_grader_result = {}
            return -0.05, "No query string provided."

        self._last_query_result = _run_query_safe(self._conn, action.query)

        if not self._last_query_result.success:
            self._last_error = self._last_query_result.error_message
            self._last_grader_result = {}
            return -0.05, f"Query error: {self._last_query_result.error_message}"

        # Run grader
        grader_module = GRADER_MODULES[self._task_id]
        if self._task_id == "task1_easy":
            result = grader_module.grade(self._conn, action.query, self._step_count, self._max_steps)
        else:
            result = grader_module.grade(
                self._conn, action.query, self._step_count,
                self._max_steps, self._cleaning_actions
            )

        # FIX: store grader result so Reward construction can use it
        self._last_grader_result = result

        score    = result["score"]
        feedback = result["feedback"]

        # step_reward == grader score (0.0 – 1.0)
        step_reward = score

        # Mark done if task fully solved
        if result.get("output_matches", False):
            self._done = True
            self._total_reward = round(max(0.001, min(0.999, score)), 4)   # pin total to exact grader score

        return step_reward, feedback

    def _handle_drop_nulls(self, action: Action):
        table = getattr(action, "table_name", None) or self._get_default_table()
        col   = action.column_name
        if not col:
            self._last_error = "column_name required for drop_nulls."
            return -0.05, "Missing column_name."
        try:
            before = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            self._conn.execute(f"DELETE FROM {table} WHERE {col} IS NULL")
            self._conn.commit()
            after = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            dropped = before - after
            self._cleaning_actions.append(f"drop_nulls:{table}.{col}")
            return 0.10, f"Dropped {dropped} null rows from {table}.{col}."
        except Exception as e:
            self._last_error = str(e)
            return -0.05, f"drop_nulls failed: {e}"

    def _handle_drop_duplicates(self, action: Action):
        table = getattr(action, "table_name", None) or self._get_default_table()
        try:
            before = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            self._conn.execute(f"""
                DELETE FROM {table}
                WHERE rowid NOT IN (
                    SELECT MIN(rowid) FROM {table}
                    GROUP BY {', '.join(self._get_column_names(table))}
                )
            """)
            self._conn.commit()
            after = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            dropped = before - after
            self._cleaning_actions.append(f"drop_duplicates:{table}")
            return 0.10, f"Removed {dropped} duplicate rows from {table}."
        except Exception as e:
            self._last_error = str(e)
            return -0.05, f"drop_duplicates failed: {e}"

    def _handle_rename_column(self, action: Action):
        table = getattr(action, "table_name", None) or self._get_default_table()
        old   = action.column_name
        new   = action.new_column_name
        if not old or not new:
            self._last_error = "column_name and new_column_name required."
            return -0.05, "Missing column name(s)."
        try:
            self._conn.execute(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}")
            self._conn.commit()
            self._cleaning_actions.append(f"rename:{table}.{old}->{new}")
            return 0.15, f"Renamed column '{old}' to '{new}' in table '{table}'."
        except Exception as e:
            self._last_error = str(e)
            return -0.05, f"rename_column failed: {e}"

    def _handle_cast_column(self, action: Action):
        table   = getattr(action, "table_name", None) or self._get_default_table()
        col     = action.column_name
        cast_to = action.cast_to
        if not col or not cast_to:
            self._last_error = "column_name and cast_to required."
            return -0.05, "Missing column_name or cast_to."
        try:
            # SQLite doesn't support ALTER COLUMN TYPE — we update in place
            self._conn.execute(f"""
                UPDATE {table}
                SET {col} = CAST({col} AS {cast_to})
                WHERE CAST({col} AS {cast_to}) IS NOT NULL
            """)
            # Remove rows where cast fails (non-numeric values stay as NULL after bad cast)
            self._conn.execute(f"""
                DELETE FROM {table}
                WHERE CAST({col} AS {cast_to}) IS NULL AND {col} IS NOT NULL
            """)
            self._conn.commit()
            self._cleaning_actions.append(f"cast:{table}.{col}->{cast_to}")
            return 0.10, f"Cast column '{col}' to {cast_to} in '{table}'. Bad rows removed."
        except Exception as e:
            self._last_error = str(e)
            return -0.05, f"cast_column failed: {e}"

    def _handle_clean_column(self, action: Action):
        table      = getattr(action, "table_name", None) or self._get_default_table()
        col        = action.column_name
        fill_value = action.fill_value
        if not col:
            self._last_error = "column_name required for clean_column."
            return -0.05, "Missing column_name."
        try:
            if fill_value is not None:
                self._conn.execute(
                    f"UPDATE {table} SET {col} = ? WHERE {col} IS NULL", (fill_value,)
                )
                msg = f"Filled nulls in '{col}' with '{fill_value}'."
            else:
                self._conn.execute(f"DELETE FROM {table} WHERE {col} IS NULL")
                msg = f"Dropped rows where '{col}' is null."
            self._conn.commit()
            self._cleaning_actions.append(f"clean:{table}.{col}")
            return 0.08, msg
        except Exception as e:
            self._last_error = str(e)
            return -0.05, f"clean_column failed: {e}"

    def _handle_done(self, action: Action):
        """Agent signals it is done without submitting a final query."""
        self._done = True
        if self._total_reward > 0:
            return 0.0, "Episode ended by agent. Final score reflects progress made."
        return -0.10, "Episode ended by agent with no progress recorded."

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_default_table(self) -> str:
        """Return the first table in the task."""
        return self._task_module.get_task_info()["tables"][0]

    def _get_column_names(self, table: str) -> List[str]:
        pragma = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        return [row[1] for row in pragma]

    def _get_all_tables(self) -> List[TableInfo]:
        tables = self._task_module.get_task_info()["tables"]
        return [_get_table_info(self._conn, t) for t in tables]

    def _build_observation(self, feedback: str = "") -> Observation:
        task_info = self._task_module.get_task_info()
        tables    = self._get_all_tables()
        return Observation(
            task_id           = self._task_id,
            difficulty        = DIFFICULTY_MAP[self._task_id],
            goal              = task_info["description"],
            instructions      = task_info["instructions"],
            tables            = tables,
            broken_query      = task_info["broken_query"],
            last_action       = self._last_action,
            last_action_error = self._last_error,
            last_query_result = self._last_query_result,
            step_count        = self._step_count,
            max_steps         = self._max_steps,
            current_score     = self._total_reward,
            done              = self._done,
            episode_id        = self._episode_id,
        )

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
