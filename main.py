from pathlib import Path
from src import indexer
from src.engine import RAGEngine

HybridIndexer = getattr(indexer, "HybridIndexer")

def main():
    vault_dir = "vault_samples"
    Path(vault_dir).mkdir(parents=True, exist_ok=True)
    
    # Ensure sample note exists
    sample_note = Path(vault_dir) / "project_architecture.md"
    if not sample_note.exists():
        sample_note.write_text(
            "# Forge Digital Solutions\n\n"
            "Forge Digital Solutions operates independently to build custom responsive web solutions for clients.\n\n"
            "## Tech Stack & Infrastructure\n"
            "The core technology stack includes Python, Polars, DuckDB, FastAPI, and Docker running on Linux/WSL2. "
            "We avoid Google Cloud Platform (GCP) entirely in development environments, preferring local-first tools.",
            encoding="utf-8"
        )

    print("--- Step 2: Initializing Persistent Index & Running Incremental Sync ---")
    indexer = HybridIndexer(db_path="vault_index.duckdb", model_name="all-MiniLM-L6-v2")
    indexer.index_vault_incrementally(vault_dir)

    print("\n--- Step 3: Initializing Local RAG Engine ---")
    rag = RAGEngine(indexer=indexer, model_name="llama3")

    print("\n--- Step 4: Executing Query ---")
    query_text = "What is the tech stack for Forge Digital Solutions and do they use GCP?"
    result = rag.query(query_text)

    print("\n================ AI RESPONSE ================")
    print(result["response"])
    print("\n================ SOURCES CITED ================")
    for src in result["sources"]:
        print(f"🔗 {src}")

if __name__ == "__main__":
    main()