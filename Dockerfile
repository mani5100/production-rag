FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy everything needed for build
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Create data folder and set ownership
RUN mkdir -p /app/data && \
    chown -R appuser:appgroup /app

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "nxb_chatbot.main:app", "--host", "0.0.0.0", "--port", "8000"]