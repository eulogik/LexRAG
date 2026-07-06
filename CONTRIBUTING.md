# Contributing to LexRAG

Thank you for your interest in LexRAG! We welcome contributions from the legal-tech and AI community.

## Code of Conduct

All contributors must abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/eulogik/LexRAG.git`
3. Create a virtual environment: `python3 -m venv venv && source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Install dev dependencies: `pip install pytest pytest-cov`
6. Run tests: `pytest`

## Development Workflow

- Create a feature branch: `git checkout -b feat/my-feature`
- Write tests for your changes
- Ensure all tests pass: `pytest`
- Run linting: `ruff check .` (if available)
- Commit with conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, etc.
- Push and open a Pull Request

## Pull Request Guidelines

- Link to any relevant issues
- Include test coverage for new functionality
- Update documentation (README, inline docs) as needed
- Ensure the PR passes CI checks

## Project Structure

```
LexRAG/
├── api/          # FastAPI server, RAG engine, memory, utils
├── embeddings/   # Vector embedding models (fastembed + Qdrant)
├── scrapers/     # Legal document scrapers (UAE, India)
├── scripts/      # Ingestion, updates, batch processing
├── ui/           # Terminal-style SPA frontend
├── tests/        # Test suite
└── data/         # SQLite DB, raw/processed documents
```

## Adding a New Scraper

1. Create `scrapers/<jurisdiction>_scraper.py`
2. Implement functions that call `scripts.ingest.ingest_text()`
3. Add the scraper to `scripts/daily_update.py`
4. Write tests for the scraper

## Adding a New LLM Provider

1. Add a streaming function in `api/rag_engine.py` (e.g., `stream_anthropic()`)
2. Register in `stream_provider()` dispatcher
3. Add the provider to `MODEL_CATALOG` in `api/main.py`
4. Add the provider label in `ui/app.js`
5. Document the required environment variable

## Questions?

Open an issue at https://github.com/eulogik/LexRAG/issues
Or reach out to engineering@eulogik.com
