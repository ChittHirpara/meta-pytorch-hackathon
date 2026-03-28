FROM python:3.11-slim

LABEL maintainer="openenv-hackathon"
LABEL description="OpenEnv SQL Repair Environment"
LABEL version="1.0.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app/           ./app/
COPY openenv.yaml   .
COPY inference.py   .
COPY README.md      .

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
