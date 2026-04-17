# Multi-stage Dockerfile
# Stage 1: Training image (heavy — includes CUDA and training deps)
# Stage 2: Inference image (slim — just what's needed to serve)

# ── Inference Stage ───────────────────────────────────────────────────────────
FROM python:3.10-slim as inference

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (no CUDA for CPU inference server)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY models/ ./models/
COPY data/processed/ ./data/processed/
COPY checkpoints/ ./checkpoints/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
