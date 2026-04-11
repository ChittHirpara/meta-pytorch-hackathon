# Use Debian slim that ships both Node 20 and allows easy Python 3 install
FROM node:20-bookworm-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python-is-python3 && \
    # Create manual symlinks as a triple-safety measure
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/python3 /usr/local/bin/python && \
    rm -rf /var/lib/apt/lists/*

# ── Environment ───────────────────────────────────────────────────────────────
ENV PORT=7860
ENV PATH="/usr/src/app:/usr/src/app/server:${PATH}"
ENV PYTHONPATH="/usr/src/app"

# ── App directory ─────────────────────────────────────────────────────────────
WORKDIR /usr/src/app

# ── Python inference deps ─────────────────────────────────────────────────────
COPY pyproject.toml ./
COPY uv.lock ./
COPY inference.py ./
COPY python ./
RUN chmod +x inference.py python && \
    pip3 install --no-cache-dir --break-system-packages .

# ── Node server deps ──────────────────────────────────────────────────────────
COPY server/package*.json ./server/
WORKDIR /usr/src/app/server
RUN npm install

# Copy all server files
COPY server/ .

# Copy openenv spec to root
WORKDIR /usr/src/app
COPY openenv.yaml .

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 7860

WORKDIR /usr/src/app/server
CMD [ "npm", "start" ]
