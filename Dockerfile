# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder

ENV POETRY_INSTALLER_MAX_WORKERS=1 \
    POETRY_HTTP_TIMEOUT=600 \
    PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=10

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV POETRY_VERSION=2.2.1

RUN curl -sSL https://install.python-poetry.org | python3 - --version $POETRY_VERSION && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry && \
    poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true

COPY pyproject.toml poetry.lock ./

RUN --mount=type=cache,target=/root/.cache/pypoetry \
    --mount=type=cache,target=/root/.cache/pip \
    poetry install --no-root --without dev --no-interaction --no-ansi

COPY . .

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd -m appuser

WORKDIR /app

COPY --chown=appuser:appuser --from=builder /app /app

USER appuser

ENV PATH="/app/.venv/bin:$PATH"