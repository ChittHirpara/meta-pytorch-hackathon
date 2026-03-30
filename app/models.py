"""
models.py — Core Pydantic models for OpenEnv SQL Repair Environment

These typed models define the contract between:
  - The environment and the agent (Observation, Action, Reward)
  - The API and any external callers
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """All actions an agent can take inside the environment."""

    SUBMIT_QUERY   = "submit_query"    # Submit a corrected SQL query
    CLEAN_COLUMN   = "clean_column"    # Remove nulls / whitespace from a column
    DROP_NULLS     = "drop_nulls"      # Drop all rows where a column is null
    DROP_DUPLICATES = "drop_duplicates" # Remove duplicate rows
    RENAME_COLUMN  = "rename_column"   # Rename a column
    CAST_COLUMN    = "cast_column"     # Cast a column to a different type
    DONE           = "done"            # Agent signals task is complete


class TaskID(str, Enum):
    """Available task difficulty levels."""
    EASY   = "task1_easy"
    MEDIUM = "task2_medium"
    HARD   = "task3_hard"


class TaskDifficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class ColumnInfo(BaseModel):
    """Describes a single column in the current table schema."""
    name: str              = Field(..., description="Column name")
    dtype: str             = Field(..., description="Data type, e.g. 'TEXT', 'INTEGER', 'REAL'")
    nullable: bool         = Field(True, description="Whether nulls are present")
    null_count: int        = Field(0, description="Number of null values in this column")
    sample_values: List[Any] = Field(default_factory=list, description="Up to 3 sample values")


class TableInfo(BaseModel):
    """Snapshot of a single table in the environment."""
    table_name: str              = Field(..., description="Name of the table")
    row_count: int               = Field(..., description="Total number of rows")
    columns: List[ColumnInfo]    = Field(..., description="Column metadata")
    preview_rows: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="First 5 rows as a list of dicts"
    )


class QueryResult(BaseModel):
    """Result of running a SQL query."""
    success: bool                = Field(..., description="Did the query run without error?")
    rows: List[Dict[str, Any]]   = Field(default_factory=list, description="Returned rows")
    row_count: int               = Field(0, description="Number of rows returned")
    error_message: Optional[str] = Field(None, description="SQL error if success=False")
    columns: List[str]           = Field(default_factory=list, description="Column names returned")


# ---------------------------------------------------------------------------
# Action — what the agent sends to the environment
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """
    Action submitted by the agent on each step.

    Examples
    --------
    Submit a corrected SQL query:
        {"action_type": "submit_query", "query": "SELECT id, name FROM users WHERE age > 18"}

    Drop nulls from a column:
        {"action_type": "drop_nulls", "column_name": "email"}

    Rename a column:
        {"action_type": "rename_column", "column_name": "usr_id", "new_column_name": "user_id"}

    Cast a column type:
        {"action_type": "cast_column", "column_name": "age", "cast_to": "INTEGER"}

    Signal done:
        {"action_type": "done"}
    """

    action_type: ActionType = Field(..., description="Type of action to perform")

    # For submit_query
    query: Optional[str] = Field(
        None, description="SQL query string (required for submit_query)"
    )

    # For column operations
    column_name: Optional[str] = Field(
        None, description="Target column name (for clean/drop/rename/cast actions)"
    )

    # For rename_column
    new_column_name: Optional[str] = Field(
        None, description="New column name (required for rename_column)"
    )

    # For cast_column
    cast_to: Optional[str] = Field(
        None, description="Target data type (required for cast_column), e.g. 'INTEGER'"
    )

    # For multi-table tasks — specifies which table to operate on
    table_name: Optional[str] = Field(
        None, description="Target table name (required for multi-table tasks like task3_hard)"
    )

    # For clean_column
    fill_value: Optional[str] = Field(
        None,
        description="Value to fill nulls with (optional for clean_column; if omitted, nulls are dropped)"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {"action_type": "submit_query", "query": "SELECT * FROM orders WHERE total > 0"},
                {"action_type": "drop_nulls", "column_name": "email"},
                {"action_type": "rename_column", "column_name": "usr_id", "new_column_name": "user_id"},
                {"action_type": "cast_column", "column_name": "price", "cast_to": "REAL"},
                {"action_type": "done"},
            ]
        }


# ---------------------------------------------------------------------------
# Observation — what the environment sends back to the agent
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    Full observation returned to the agent after reset() or step().

    Contains everything the agent needs to understand the current state
    and decide the next action.
    """

    # Task context
    task_id: str              = Field(..., description="Current task identifier")
    difficulty: TaskDifficulty = Field(..., description="Task difficulty level")
    goal: str                 = Field(..., description="Natural language description of the task goal")
    instructions: str         = Field(..., description="Step-by-step hints about available actions")

    # Current database state
    tables: List[TableInfo]   = Field(..., description="All tables in the current environment")
    broken_query: Optional[str] = Field(
        None, description="The SQL query the agent needs to fix (if applicable)"
    )

    # Feedback from last action
    last_action: Optional[str]       = Field(None, description="The last action taken by the agent")
    last_action_error: Optional[str] = Field(None, description="Error from last action, if any")
    last_query_result: Optional[QueryResult] = Field(
        None, description="Result of last submitted query"
    )

    # Progress tracking
    step_count: int           = Field(0, description="Number of steps taken so far")
    max_steps: int            = Field(10, description="Maximum steps allowed in this episode")
    current_score: float      = Field(0.0, description="Running score so far (0.0–1.0)")

    # Episode status
    done: bool                = Field(False, description="Whether the episode is complete")
    episode_id: str           = Field(..., description="Unique episode identifier")


# ---------------------------------------------------------------------------
# Reward — scoring for each step
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """
    Reward signal returned after each step.

    Provides dense feedback so the agent can learn from every action,
    not just at the end of the episode.
    """

    # Core score
    step_reward: float  = Field(..., ge=-1.0, le=1.0,
                               description="Reward for this step only (-1.0 to +1.0)")
    total_reward: float = Field(..., ge=0.0,  le=1.0,
                               description="Cumulative reward this episode (0.0 to 1.0)")

    # Breakdown for interpretability
    correctness_score: float   = Field(0.0, ge=0.0, le=1.0,
                                       description="How correct the query output is")
    data_quality_score: float  = Field(0.0, ge=0.0, le=1.0,
                                       description="How clean the data is")
    efficiency_penalty: float  = Field(0.0, ge=0.0, le=1.0,
                                       description="Penalty for wasted steps")

    # Human-readable feedback
    feedback: str              = Field(..., description="Natural language explanation of the reward")

    # Milestone flags
    query_executes: bool       = Field(False, description="True if the submitted query runs without error")
    schema_correct: bool       = Field(False, description="True if schema matches expected")
    output_matches: bool       = Field(False, description="True if query output matches ground truth exactly")
    task_complete: bool        = Field(False, description="True if task is fully solved")


# ---------------------------------------------------------------------------
# StepResult — combined return of step()
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Full return value of env.step(action)."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# State — internal snapshot (for GET /state)
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    """
    Internal state snapshot — returned by GET /state.
    Useful for debugging and monitoring.
    """
    episode_id: str
    task_id: str
    difficulty: TaskDifficulty
    step_count: int
    max_steps: int
    total_reward: float
    done: bool
    tables: List[TableInfo]
    broken_query: Optional[str]
    action_history: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Task metadata (for openenv.yaml and /tasks endpoint)
# ---------------------------------------------------------------------------

class TaskInfo(BaseModel):
    """Metadata about a single task."""
    task_id: str
    name: str
    difficulty: TaskDifficulty
    description: str
    max_steps: int
    max_score: float = 1.0
    action_types_used: List[str]


class EnvironmentInfo(BaseModel):
    """Top-level environment metadata."""
    name: str           = "SQL Repair Environment"
    version: str        = "1.0.0"
    description: str    = (
        "An OpenEnv environment where agents learn to fix broken SQL queries "
        "and clean dirty tabular data across three difficulty levels."
    )
    tasks: List[TaskInfo]
    supported_actions: List[str] = [a.value for a in ActionType]
