# SQL Repair Environment

This project implements a SQL repair environment for training and evaluating LLMs.

## Project Structure
- `app/`: Core FastAPI application and logic.
- `tasks/`: Individual SQL repair tasks (Easy, Medium, Hard).
- `graders/`: Logic for grading the responses for each task.
- `tests/`: Unit tests for the environment.

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `uvicorn app.main:app --reload`
