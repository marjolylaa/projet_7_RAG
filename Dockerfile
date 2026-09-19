FROM python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" \
    PORT=8000 \
    HOST=0.0.0.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen --no-install-project

COPY main.py ./
COPY src/__init__.py ./src/__init__.py
COPY src/chatbot.py ./src/chatbot.py
COPY src/api.py ./src/api.py
COPY src/static/ui-chatbot.html ./src/static/ui-chatbot.html
COPY src/mon_index_langchain_evenements_hnsw_rapide/index.faiss ./src/mon_index_langchain_evenements_hnsw_rapide/index.faiss
COPY src/mon_index_langchain_evenements_hnsw_rapide/index.pkl ./src/mon_index_langchain_evenements_hnsw_rapide/index.pkl

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
