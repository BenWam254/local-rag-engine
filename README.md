# Local Hybrid RAG Engine
An enterprise-grade, fully offline Retrieval-Augmented Generation (RAG) architecture built to turn local markdown vaults into an interactive, real-time AI knowledge base.

This engine is designed with privacy and performance in mind: no cloud APIs, no external telemetry, and no redundant indexing. It operates as a headless FastAPI microservice that instantly synchronizes with file system events and queries a local DuckDB vector store.

## Features

* **Real-Time Incremental Indexing:** An asynchronous daemon (`watchfiles`) monitors your local directories (e.g., Obsidian vaults) and incrementally syncs modifications to the vector database the exact second a file is saved.
* **Persistent Vector Storage:** Built on DuckDB for zero-dependency, lightning-fast persistence. Bypasses redundant re-embedding by utilizing MD5 file hashing and caching.
* **Hybrid Retrieval Pipeline:** Casts a wide net using Dense Vector Search (`all-MiniLM-L6-v2`) and BM25, combined mathematically via Reciprocal Rank Fusion (RRF).
* **Surgical Precision Reranking:** Filters and scores the hybrid candidate pool using a secondary AI model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) before generating the final response.
* **Headless FastAPI Microservice:** Keeps the language model and database hot in memory, exposing a clean REST API for frontends or CLI clients.
* **Fully Grounded Generation:** Leverages local LLMs (via Ollama) to synthesize answers with strict, exact-file citations to prevent hallucination.

## Tech Stack

* **Backend:** Python, FastAPI, Uvicorn
* **Database:** DuckDB (Persistent Blob & Text Storage)
* **Embedding & Reranking:** `sentence-transformers`, PyTorch
* **Inference:** Ollama (`llama3`)
* **Event Watcher:** `watchfiles`
* **Package Management:** `uv`

## Project Structure

```text
├── src/
│   ├── engine.py       # Core RAG logic, prompting, and LLM generation
│   ├── indexer.py      # DuckDB schema, BM25, dense embeddings, and Cross-Encoder
│   └── parser.py       # Semantic markdown chunking
├── vault_samples/      # Local directory for markdown files (e.g., Obsidian Vault)
├── server.py           # FastAPI microservice
├── watcher.py          # Real-time background file system daemon
├── chat.py             # Interactive CLI chat interface
└── main.py             # Sandbox script for sequential pipeline testing
```

## Getting Started

1. ### Prerequisites
Install Ollama and pull your preferres model:

``` bash
ollama run llama 3
```

2. ### Installation
Clone the repository and install the dependencies:

``` bash
git clone [https://github.com/BenWam254/local-rag-engine.git](https://github.com/BenWam254/local-rag-engine.git)
cd local-rag-engine
uv pip install fastapi uvicorn requests watchfiles sentence-transformers duckdb rank-bm25
```

3. ### Running the Microservice Architecture
This engine is designed to run completely decoupled. Open three separate terminal windows to spin up the architecture:



#### Terminal 1: Start the API Server
Keeps the engine and LLM hot in memory.

```bash
uvicorn server:app --host 127.0.0.1 --port 8000
```

#### Terminal 2: Start the Vault Watcher
Listens for local file changes and syncs them to DuckDB incrementally.

```bash
python watcher.py
```

#### Terminal 3: Launch the Chat Client

```bash
python chat.py
```

## Future Roadmap
* [] Connect directly to absolute paths for extensice Obsidian vault indexing.
* [] Build external data ingestion pipelines (PDFs, local web scraping).
* [] Develop a custom web UI for client-facing deployments.
