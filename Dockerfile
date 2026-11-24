# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

WORKDIR /build

# Install curl to fetch uv
RUN apt-get update && apt-get install --no-install-recommends -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for ultra-fast dependency installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install production serving and data dependencies into standalone virtual environment
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python \
    fastapi uvicorn pydantic python-dotenv requests httpx pandas "numpy<2.4"

# --- Runtime Stage ---
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install curl for container healthchecks
RUN apt-get update && apt-get install --no-install-recommends -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Copy application source code, models, configs, and data
COPY --chown=appuser:appuser src /app/src
COPY --chown=appuser:appuser config /app/config
COPY --chown=appuser:appuser models /app/models
COPY --chown=appuser:appuser data /app/data

EXPOSE 8000

# Healthcheck probe
HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production server command with 4 uvicorn worker processes
CMD ["uvicorn", "dengue_mlops.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
