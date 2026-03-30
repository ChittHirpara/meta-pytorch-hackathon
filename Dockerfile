# ── Base image ────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Metadata ─────────────────────────────────────────────────────────────
LABEL maintainer="openenv-hackathon"
LABEL description="OpenEnv SQL Repair Environment"
LABEL version="1.0.0"

# ── System dependencies ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────
COPY pyproject.toml README.md ./
COPY app/ ./app/
COPY server/ ./server/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir . \
    && pip install --no-cache-dir openenv-core>=0.2.0 \
    && python -m uvicorn --version

COPY openenv.yaml .
COPY inference.py .

# ── Create non-root user (HF Spaces requirement) ──────────────────────────
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# ── Expose port (HF Spaces default) ──────────────────────────────────────
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')"

# ── Start server ──────────────────────────────────────────────────────────
# ── Start server ──────────────────────────────────────────────────────────
# Using 'python -m uvicorn' is the most reliable way to ensure the uvicorn 
# module is found within the Python environment, avoiding PATH issues.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
