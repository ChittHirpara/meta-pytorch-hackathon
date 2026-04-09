"""
main.py — FastAPI application for OpenEnv SQL Repair Environment

Endpoints:
  GET  /              → health check
  GET  /info          → environment metadata
  GET  /tasks         → list all tasks
  POST /reset         → start new episode
  POST /step          → take an action
  GET  /state         → current environment state
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.environment import SQLRepairEnvironment
from app.models import (
    Action, EnvironmentInfo, EnvironmentState,
    Observation, StepResult, TaskDifficulty, TaskInfo,
)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

# One global environment instance (single-user for hackathon scope)
_env: Optional[SQLRepairEnvironment] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _env
    _env = SQLRepairEnvironment()
    print("[Success] SQL Repair Environment initialized.")
    yield
    if _env:
        _env.close()
    print("[Closed] Environment closed.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OpenEnv — SQL Repair Environment",
    description=(
        "An OpenEnv-compliant environment where AI agents learn to fix broken SQL queries "
        "and clean dirty tabular data across three difficulty levels."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def get_env() -> SQLRepairEnvironment:
    if _env is None:
        raise HTTPException(status_code=503, detail="Environment not initialized.")
    return _env


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", summary="Health check")
def health_check():
    """Returns 200 when the server is running."""
    return {"status": "ok", "environment": "SQL Repair Environment", "version": "1.0.0"}


@app.get("/info", response_model=EnvironmentInfo, summary="Environment metadata")
def get_info():
    """Return top-level environment metadata."""
    tasks = [
        TaskInfo(
            task_id="task1_easy",
            name="Fix SQL Syntax Error",
            difficulty=TaskDifficulty.EASY,
            description="Fix a SQL query with a typo/syntax error against a clean employees table.",
            max_steps=5,
            action_types_used=["submit_query", "done"],
        ),
        TaskInfo(
            task_id="task2_medium",
            name="Clean Data and Fix Query Logic",
            difficulty=TaskDifficulty.MEDIUM,
            description="Clean dirty orders data (nulls, dupes, wrong types) then fix a logic bug in the query.",
            max_steps=10,
            action_types_used=["drop_nulls", "drop_duplicates", "rename_column", "cast_column", "submit_query", "done"],
        ),
        TaskInfo(
            task_id="task3_hard",
            name="Multi-Table Join Repair",
            difficulty=TaskDifficulty.HARD,
            description="Clean 3 tables and rewrite a broken multi-table JOIN + aggregation query.",
            max_steps=15,
            action_types_used=["drop_nulls", "drop_duplicates", "rename_column", "cast_column", "submit_query", "done"],
        ),
    ]
    return EnvironmentInfo(tasks=tasks)


@app.get("/tasks", summary="List all tasks")
def list_tasks():
    """List all available tasks with descriptions."""
    return {
        "tasks": [
            {
                "task_id": "task1_easy",
                "name": "Fix SQL Syntax Error",
                "difficulty": "easy",
                "max_steps": 5,
                "description": "Fix typo/syntax in a SQL query against the employees table.",
            },
            {
                "task_id": "task2_medium",
                "name": "Clean Data and Fix Query Logic",
                "difficulty": "medium",
                "max_steps": 10,
                "description": "Clean dirty orders data and fix logical errors in the SQL query.",
            },
            {
                "task_id": "task3_hard",
                "name": "Multi-Table Join Repair",
                "difficulty": "hard",
                "max_steps": 15,
                "description": "Fix schema, clean 3 tables, rewrite a broken JOIN + aggregation query.",
            },
        ]
    }


class ResetRequest(BaseModel):
    """Optional JSON body for /reset."""
    task_id: Optional[str] = "task1_easy"


@app.post("/reset", response_model=Observation, summary="Start a new episode")
async def reset(
    request: Request,
    task_id: Optional[str] = Query(
        default=None,
        description="Task to run: task1_easy | task2_medium | task3_hard",
    ),
):
    """
    Reset the environment and start a fresh episode.

    Accepts task_id from:
      - Query param:  POST /reset?task_id=task1_easy
      - JSON body:    POST /reset  {"task_id": "task1_easy"}
      - Plain POST:   POST /reset  (defaults to task1_easy)

    Returns an Observation containing the database state, broken query, and goal.
    """
    env = get_env()

    # Try to get task_id from JSON body if not in query param
    resolved_task_id = task_id
    if resolved_task_id is None:
        try:
            body = await request.json()
            resolved_task_id = body.get("task_id", "task1_easy")
        except Exception:
            resolved_task_id = "task1_easy"

    if not resolved_task_id:
        resolved_task_id = "task1_easy"

    try:
        obs = env.reset(task_id=resolved_task_id)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult, summary="Take an action")
def step(action: Action):
    """
    Submit an action to the environment.

    The agent can:
    - `submit_query`   — provide a corrected SQL query
    - `drop_nulls`     — drop rows with null in a column
    - `drop_duplicates`— remove duplicate rows
    - `rename_column`  — rename a column
    - `cast_column`    — cast a column to a different type
    - `clean_column`   — fill or drop nulls in a column
    - `done`           — signal the agent is finished

    Returns **StepResult** with new Observation, Reward, and done flag.
    """
    env = get_env()
    try:
        result = env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.get("/state", response_model=EnvironmentState, summary="Get current state")
def state():
    """
    Return the current internal state of the environment.

    Useful for debugging, monitoring, and external validators.
    """
    env = get_env()
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/validate", summary="OpenEnv spec validation ping")
def validate():
    """Endpoint used by openenv validate tool."""
    env = get_env()
    obs = env.reset(task_id="task1_easy")
    return {
        "valid": True,
        "tasks": ["task1_easy", "task2_medium", "task3_hard"],
        "endpoints": ["/reset", "/step", "/state"],
        "episode_id": obs.episode_id,
    }
