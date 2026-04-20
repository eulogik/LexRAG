# ⚖️ LexRAG — The Legal Intelligence Terminal (v3.2.2)

LexRAG is a professional-grade, high-performance legal research platform designed by Evolucent AI. It provides institutional-level accuracy for **UAE and Indian law, taxation, accounting standards, and case law**.

Unlike traditional AI chat, LexRAG 3.2.2 uses a **Hybrid RAG Terminal** architecture that combines dense semantic search, sparse keywords, and neural reranking to ensure that every answer is grounded in authentic legal statutes.

## 🚀 Key Innovations

- **The Intelligence Terminal**: A zero-latency, minimal SPA (Single Page Application) built with Vanilla JS and modern CSS. No heavy frameworks, just pure performance.
- **Reliable SSE Streaming**: Powered by a custom FastAPI `StreamingResponse` layer with automated heartbeats to prevent connection stalls.
- **Hybrid RAG Engine**:
    - **Dense**: `intfloat/multilingual-e5-large` for semantic meaning.
    - **Sparse**: `Splade_PP_en_v1` for keyword precision (Acts, Section numbers).
    - **Reranking**: `BAAI/bge-reranker-base` for extreme relevance.
- **Jurisdiction Auto-Detection**: Intelligently switches between India and UAE law based on query keywords, or allows manual override.
- **Dynamic Model Registry**: Granular control over Groq, OpenRouter, and Ollama models, including the ability to add and persist custom model IDs.

## 🛠️ Architecture

- **Frontend**: Vanilla HTML5/JS/CSS (Atomic Design).
- **Backend**: Python 3.11, FastAPI (Asynchronous).
- **In-Memory Store**: Qdrant (On-disk native mode) + SQLite (Sessions).
- **Intelligence Layer**: RAG Pipeline with Thinking-filter (for models like Qwen).

## ⚙️ Quick Start

### 1. Installation
```bash
git clone <repository-url>
cd LexRAG
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory:
```env
# API Keys
GROQ_API_KEY=your_key
OPENROUTER_API_KEY=your_key
INDIANKANOON_TOKEN=your_key

# Defaults
LLM_PROVIDER=groq
```

### 3. Run the Terminal
```bash
# Start the FastAPI server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Visit [http://localhost:8000](http://localhost:8000) to begin.

## 📁 Project Structure

- `api/`: Core logic and API endpoints.
    - `rag_engine.py`: The "Brain" of LexRAG (Retrieval + LLM).
    - `memory.py`: Database session and history management.
- `ui/`: The high-aesthetic frontend "Terminal".
- `embeddings/`: Vectorization and Hybrid Search implementation.
- `scripts/`: Ingestion and scraping pipeline.
- `data/`: Local vector storage and statutes.

## 🛡️ Trust & Safety
LexRAG is an AI research assistant. While it uses official statutes, it does not constitute legal or financial advice. Always verify with the primary sources linked in the LexRAG citations.

---
**By Evolucent AI**
