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
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir openenv-core>=0.2.0 || true

# ── Copy project files ────────────────────────────────────────────────────
COPY app/           ./app/
COPY openenv.yaml   .
COPY inference.py   .
COPY README.md      .

# ── Create non-root user (HF Spaces requirement) ──────────────────────────
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# ── Expose port (HF Spaces default) ──────────────────────────────────────
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')"

# ── Start server ──────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
