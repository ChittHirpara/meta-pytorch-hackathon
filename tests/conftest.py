# tests/conftest.py
# Ensure the repo root is on sys.path so `app.*` imports resolve correctly.
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
