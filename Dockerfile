FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY api/ ./api/
COPY embeddings/ ./embeddings/
COPY scripts/ ./scripts/
COPY scrapers/ ./scrapers/
COPY ui/ ./ui/
COPY marketing/ ./marketing/
COPY data/statutes/ ./data/statutes/
COPY pyproject.toml README.md ./

RUN mkdir -p data qdrant_storage embeddings_cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
