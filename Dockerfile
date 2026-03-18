# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install dependencies into a prefix so they can be copied cleanly
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Create a non-root user and transfer ownership
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Use exec form so signals (SIGTERM) are forwarded correctly
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
