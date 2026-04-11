#!/usr/bin/env python3
import os
import sys

# Diagnostic logging for Phase 2 troubleshooting
print(f"--- [INIT] NetPulse INFERENCE ---", flush=True)
print(f"Python Executable: {sys.executable}", flush=True)
print(f"Python Version: {sys.version}", flush=True)
print(f"CWD: {os.getcwd()}", flush=True)
print(f"PATH: {os.environ.get('PATH')}", flush=True)
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}", flush=True)

import json
import time
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

# Required Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# If using from_docker_image(), but we use direct REST
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# NetPulse server URL
ENV_BASE_URL = os.getenv("TRIAGE_BASE_URL", "http://localhost:7860")
BENCHMARK = "netpulse"

TASKS = [
    "easy_signal_noise",
    "medium_hidden_dependency",
    "hard_multi_incident"
]

def get(url):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def post(url, data):
    r = requests.post(url, json=data, timeout=20)
    r.raise_for_status()
    return r.json()

def run_task(task_name, client, max_steps=20):
    # Log EXACTLY as required by the regex
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)
    
    try:
        reset_data = post(f"{ENV_BASE_URL}/reset", {"task_name": task_name})
        obs = reset_data["observation"]
    except Exception as e:
        print(f"[END] success=false steps=0 rewards=", flush=True)
        return False, 0
    
    rewards = []
    success = False
    
    system_prompt = """
    You are an expert Site Reliability Engineer (SRE).
    Action space expects exactly one JSON object: {"action": "string", "target": "string"}.
    Available actions: noop, inspect_service, inspect_dependency, restart_service, throttle_queue, rollback_deploy, scale_service.
    Target must be the exact name of an active service.
    """
    
    steps_taken = 0
    for i in range(1, max_steps + 1):
        steps_taken = i
        user_prompt = f"Observation:\n{json.dumps(obs, indent=2)}\n\nWhat is your next action JSON?"
        
        error_msg = "null"
        try:
            print(f">>> [STEP {i}] Requesting action from LLM (timeout=30s)...", flush=True)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={ "type": "json_object" },
                temperature=0.2,
                timeout=30.0
            )
            raw_content = response.choices[0].message.content
            if not raw_content:
                 raise ValueError("Empty response from LLM")
            action_data = json.loads(raw_content)
            
            action_payload = {"action": action_data.get("action", "noop")}
            if "target" in action_data and action_data["target"]:
                action_payload["target"] = action_data["target"]
                
        except Exception as e:
            # Fix: Avoid backslash in f-string expression for older Python versions
            clean_err = str(e).replace('\n', ' ')
            error_msg = f"LLM_OR_PARSE_ERROR: {clean_err}"
            action_payload = {"action": "noop"}
            
        action_str = json.dumps(action_payload, separators=(',', ':'))
        
        try:
            res = post(f"{ENV_BASE_URL}/step", action_payload)
            obs = res.get("observation", obs)
            done = res.get("done", False)
            success = res.get("success", False)
            reward_val = res.get("reward", 0.0)
            
            if "error" in res and res["error"]:
                 error_msg = str(res["error"]).replace('\n', ' ')
        except Exception as e:
            reward_val = 0.0
            done = True
            error_msg = str(e).replace('\n', ' ')
            
        rewards.append(reward_val)
        
        # Log EXACTLY as formatting spec dictates
        done_str = "true" if done else "false"
        print(f"[STEP] step={i} action={action_str} reward={reward_val:.2f} done={done_str} error={error_msg}", flush=True)
        
        if done:
            break
            
        time.sleep(0.5)
        
    success_str = "true" if success else "false"
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={success_str} steps={steps_taken} rewards={rewards_str}", flush=True)
    return success

def main():
    # Priority: 1. API_KEY (LiteLLM) 2. OPENAI_API_KEY 3. HF_TOKEN
    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
    base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    
    if not api_key:
        print("[WARNING] No valid API key found. Using dummy_key for local dry-run.", flush=True)
        api_key = "dummy_key"

    print(f"Initializing OpenAI client with base_url: {base_url}", flush=True)
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize OpenAI client: {e}", flush=True)
        return
    
    print(f"Connecting to NetPulse server at {ENV_BASE_URL}...", flush=True)
    health_check_passed = False
    for i in range(5):
        try:
            get(f"{ENV_BASE_URL}/health")
            health_check_passed = True
            break
        except Exception as e:
            print(f"Retry {i+1}/5: Failed to connect to NetPulse server: {e}", flush=True)
            time.sleep(2)
    
    if not health_check_passed:
        print(f"[ERROR] Could not reach NetPulse server at {ENV_BASE_URL}. Exiting.", flush=True)
        return

    print(f"Starting inference for tasks: {TASKS}", flush=True)
    for task in TASKS:
        try:
            run_task(task, client, max_steps=15)
        except Exception as e:
            print(f"[ERROR] Unhandled exception in task {task}: {e}", flush=True)
        
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] Unhandled inference exception: {e}", file=sys.stderr)
        sys.exit(1)
