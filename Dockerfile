FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies (including curl for healthcheck)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install supervisor to run both processes
RUN pip install supervisor

COPY . .

# Create data directories and set permissions
RUN mkdir -p data/uploads data/processed && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy supervisor config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

USER appuser

EXPOSE 8000 8501

# Health check (checks backend)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start supervisor (runs both backend and frontend)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]