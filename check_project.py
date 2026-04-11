"""Final project health check — runs all critical validations."""
import sys, sqlite3
sys.path.insert(0, '.')

PASS = []
FAIL = []

def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS: {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL: {name}  {detail}")

print("\n========== PROJECT HEALTH CHECK ==========\n")

# -- 1. Imports
print("[1] Core imports")
try:
    from app.graders import grader_easy, grader_medium, grader_hard
    from app.tasks import task1_easy, task2_medium, task3_hard
    from app.environment import SQLRepairEnvironment
    check("All imports OK", True)
except Exception as e:
    check("All imports OK", False, str(e))

# -- 2. Grader score bounds
print("\n[2] Grader score bounds (must be strictly 0 < score < 1)")

def test_bounds(label, grade_fn, conn, *args):
    result = grade_fn(conn, *args)
    s = result["score"]
    check(label, 0.0 < s < 1.0, f"score={s}")
    return s

conn = sqlite3.connect(":memory:")
task1_easy.setup_database(conn)
test_bounds("grader_easy (bad query)", grader_easy.grade, conn, "SELECT 1", 1, 5)
test_bounds("grader_easy (good query)", grader_easy.grade, conn,
    "SELECT id, name, department FROM employees WHERE salary > 50000", 1, 5)
conn.close()

conn = sqlite3.connect(":memory:")
task2_medium.setup_database(conn)
test_bounds("grader_medium (bad query)", grader_medium.grade, conn, "SELECT 1", 1, 10, [])
conn.close()

conn = sqlite3.connect(":memory:")
task3_hard.setup_database(conn)
test_bounds("grader_hard (bad query)", grader_hard.grade, conn, "SELECT 1", 1, 15, [])
conn.close()

# -- 3. Environment score bounds
print("\n[3] Environment score bounds")
env = SQLRepairEnvironment()
obs = env.reset("task1_easy")
check("reset() returns Observation", obs is not None)

from app.models import Action, ActionType
action = Action(action_type=ActionType.SUBMIT_QUERY,
    query="SELECT id, name, department FROM employees WHERE salary > 50000")
result = env.step(action)
s = result.reward.total_reward
check("env.step total_reward strictly (0,1)", 0.0 < s < 1.0, f"got {s}")
env.close()

# -- 4. Structured output tags
print("\n[4] inference.py structured output tags")
import subprocess, re
out = subprocess.run(["python", "inference.py", "--task", "task1_easy"],
    capture_output=True, text=True, timeout=60)
stdout = out.stdout
check("[START] tag present", "[START]" in stdout)
check("[STEP] tag present", "[STEP]" in stdout)
check("[END] tag present", "[END]" in stdout)
scores = re.findall(r"\[END\] score=([0-9.]+)", stdout)
for sc in scores:
    f = float(sc)
    check(f"[END] score={sc} strictly (0,1)", 0.0 < f < 1.0, f"score={f}")

# -- 5. server/app.py entry point
print("\n[5] server/app.py entry point")
from server.app import main
check("server.app:main callable", callable(main))

# -- 6. pyproject.toml checks
print("\n[6] pyproject.toml")
import tomllib
with open("pyproject.toml", "rb") as f:
    toml = tomllib.load(f)
deps = toml["project"]["dependencies"]
scripts = toml["project"]["scripts"]
check("openenv-core dependency present", any("openenv" in d for d in deps))
check("[project.scripts] server entry exists", "server" in scripts)
check("server entry points to server.app:main", scripts.get("server","") == "server.app:main")

# -- Summary
print(f"\n{'='*42}")
print(f"  PASSED: {len(PASS)}   FAILED: {len(FAIL)}")
if FAIL:
    print("\n  FAILING CHECKS:")
    for f in FAIL:
        print(f"    FAIL: {f}")
else:
    print("\n  ALL CHECKS PASSED - project is ready!")
print("="*42)
