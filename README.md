# ⚖️ LexRAG — Legal & Accounting AI

LexRAG is a specialized Retrieval-Augmented Generation (RAG) platform designed for professionals navigating **UAE and Indian law, taxation, accounting standards, and case law**. It provides sourced, expert-level answers by grounding LLM responses in a curated vector knowledge base of official statutes, rulings, and case documents.

## 🚀 Features

-   **Dual-Jurisdiction Intelligence**: Specialized context handling for UAE (FTA/MOJ) and India (GST/Income Tax).
-   **Multi-Provider LLM Support**: Seamlessly switch between **OpenRouter** (Gemma 31B), **Groq** (Qwen 32B), and **Local Ollama** (Qwen 14B).
-   **Deep Legal Ingestion**:
    -   **Automated Scrapers**: Daily updates from UAE Federal Tax Authority and India's CBIC.
    -   **Indian Kanoon Integration**: Direct API access to Indian case law.
    -   **Bulk PDF Ingest**: Single-command ingestion of custom legal libraries.
-   **High-Aesthetic UI**: Modern Streamlit interface with source transparency and score tracking.

## 🛠️ Tech Stack

-   **Backend**: Python 3.11, FastAPI
-   **Vector DB**: Qdrant (Vector Engine)
-   **Embeddings**: `intfloat/multilingual-e5-base` (Multilingual Support)
-   **UI**: Streamlit
-   **Orchestration**: Custom Python RAG Pipeline

## ⚙️ Setup Instructions

### 1. Prerequisites
- Docker (for Qdrant)
- Ollama (for local inference, optional)
- Python 3.10+

### 2. Installation
```bash
git clone <repository-url>
cd LexRAG
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt # or refer to plan.md for manual install
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
INDIANKANOON_TOKEN=your_token_here
GROQ_API_KEY=your_groq_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
LLM_PROVIDER=openrouter  # options: openrouter, groq, ollama
```

### 4. Running the System
**Start Qdrant (Docker):**
```bash
docker run -d --name qdrant -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

**Launch Backend & UI:**
- **UI**: `streamlit run ui/app.py`
- **Backend API**: `uvicorn api.main:app --reload`
- **Daily Scraper**: `python3 scripts/daily_update.py`

## 📂 Project Structure

-   `api/`: FastAPI endpoints and the core RAG reasoning engine.
-   `embeddings/`: Logic for document vectorization and Qdrant search.
-   `scrapers/`: Automated crawlers for legal portals.
-   `scripts/`: Utility scripts for ingestion and scheduling.
-   `ui/`: Streamlit frontend implementation.
-   `data/`: Local storage for raw and processed documents.

## ⚖️ Disclaimer
*LexRAG is an AI-powered research assistant and does not constitute legal or financial advice. Always consult with a qualified professional for critical legal matters.*
