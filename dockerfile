FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.30 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    HF_HOME=/app/hf_home \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 1) Dependencies only (slow, cached until pyproject.toml or uv.lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) Embedding model, downloaded at build time (cached until step 1 changes)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# 3) Our code and policy (fast; rebuilt on every code change)
COPY src ./src
COPY policies ./policies
RUN uv sync --frozen --no-dev

# 4) Run as a normal user, fully offline
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/.cache \
    && chown appuser:appuser /app/.cache
ENV HF_HUB_OFFLINE=1
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# One worker only: the busy lock and the model live inside one process.
CMD ["uvicorn", "--factory", "contractguard.api.app:create_app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]