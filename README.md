# OpenEnv SQL Repair Environment

An OpenEnv-compliant reinforcement learning environment where AI agents learn to clean tabular data and fix broken SQL queries. 

## Motivation & Real-World Utility
Data Engineers and backend developers spend an immense amount of time debugging SQL queries and cleaning dirty data (handling nulls, deduping rows, verifying schema types). This environment models a genuine, day-to-day real-world scenario where an agent must inspect a database, apply data cleaning operations, and successfully submit a valid SQL query that matches the exact desired business metric. This fills a significant gap in evaluating how frontier models reason with code dependencies and data structures sequentially.

## Setup Instructions

This environment is fully containerized and compatible with Hugging Face spaces.

**Running via Docker:**
```bash
docker build -t openenv-sql-repair .
docker run -p 7860:7860 openenv-sql-repair
```

**Running Locally (Python):**
```bash
pip install -r requirements.txt
uvicorn app.main:app --port 7860
```

## Running the Baseline Agent

We provide a robust baseline inference script using the OpenAI client (which is compatible with any OpenAI-compatible endpoint, including Gemini or open weights). It features dynamic prompt switching and Chain-of-Thought reasoning.

```bash
# Set your environment variables
export OPENAI_API_KEY="your-api-key"
export API_BASE_URL="https://api.openai.com/v1"   # Or your compatible endpoint
export MODEL_NAME="gpt-4o-mini"
export ENV_BASE_URL="http://localhost:7860"

# Run the inference script
python inference.py --task all
```

## Observation & Action Spaces

### Observation Space
The observation space is a fully typed, structured JSON object containing everything the agent needs to understand the environment state:
* `task_id` and `difficulty`
* `goal` & `instructions` (What the agent must achieve)
* `broken_query` (The SQL that needs fixing)
* `tables`: Complete schema information including `table_name`, `row_count`, and column details (`dtype`, `null_count`, sample values).
* `last_query_result` & `last_action_error`: Feedback from the previous step.

### Action Space (Discrete)
The agent interacts with the environment by executing specific, typed JSON actions:
* `drop_nulls`: Drop rows where a specific column is null.
* `drop_duplicates`: Remove duplicate rows from a table.
* `rename_column`: Rename a table column to fix schema drift.
* `cast_column`: Cast a column to the correct data type (e.g., TEXT to REAL).
* `clean_column`: Fill missing values.
* `submit_query`: Submit the final repaired SQL query for evaluation.
* `done`: Signal episode completely finished.

## Tasks & Expected Difficulty

We provide 3 distinct tasks that gradually challenge the agent's context window and planning capabilities:

1. **Fix SQL Syntax Error (Easy)**
   * **Description**: A single clean table. The agent simply needs to diagnose a typo in a SELECT query and fix it to return the right rows.
   * **Max Steps**: 5
2. **Clean Data and Fix Query Logic (Medium)**
   * **Description**: A messy orders table with null values, duplicates, and incorrect data types. The query itself has faulty logic. The agent must first drop bad rows, rename columns, and cast types, before fixing the query logic.
   * **Max Steps**: 10
3. **Multi-Table Join Repair (Hard)**
   * **Description**: Three separate tables (customers, products, transactions) full of missing keys and duplicates. The required SQL query involves joining all three tables and performing an aggregate. The agent must systematically clean all tables first, then orchestrate a complex SQL fix.
   * **Max Steps**: 15

## Baseline Scores
Running our baseline agent script over all 3 tasks with `gemini-2.0-flash` achieves the following average performance out-of-the-box (0.0 to 1.0 scale):

| Task | Final Score (Reward) | Agent Steps Taken | 
|---|---|---|
| `task1_easy` | **1.000** | 1 | 
| `task2_medium` | **0.500 - 0.700** | 4-8 | 
| `task3_hard` | **0.800+** | 8-12 | 

*(Note: Actual multi-phase performance relies heavily on rate limits and exact CoT trajectory. Task 3 is notoriously difficult and requires careful multi-table reasoning).*
