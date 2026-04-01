# ── Base image ───────────────────────────────────────────────────────────────
# Use slim Debian-based Python 3.12 — small footprint, no conda overhead.
FROM python:3.12-slim

# ── System dependencies ───────────────────────────────────────────────────────
# tzdata  — needed for zoneinfo ("America/New_York") to work correctly
# ca-certificates — needed for HTTPS calls to Alpaca
RUN apt-get update && apt-get install -y --no-install-recommends \
        tzdata \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker caches this layer — only rebuilt when
# requirements.txt changes, not on every code change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY config/ config/
COPY data/   data/
COPY trader/ trader/
COPY main.py .

# ── Environment defaults ──────────────────────────────────────────────────────
# These are overridden at runtime by Google Secret Manager / --env flags.
# Never put real keys here.
ENV ALPACA_API_KEY=""
ENV ALPACA_SECRET_KEY=""
ENV ALPACA_PAPER="true"

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Runs the V6 live trader. stdout/stderr are captured by Cloud Logging.
CMD ["python", "-u", "main.py"]
