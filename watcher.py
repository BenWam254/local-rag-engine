from pathlib import Path
from watchfiles import watch
from src.indexer import HybridIndexer

def start_watcher(vault_dir: str = "vault_samples"):
    # Resolve to an absolute path for reliable Windows file event tracking
    target_path = Path(vault_dir).resolve()
    print(f" Starting real-time vault watcher on directory: '{target_path}'...")

    indexer = HybridIndexer(db_path="vault_index.duckdb", model_name="all-MiniLM-L6-v2")

    print("\n Watcher active. Waiting for file changes... (Press Ctrl+C to stop)")

    for changes in watch(target_path):
        print(f"\n File system event detected: {changes}")
        print("Syncing changes to DuckDB...")

        indexer.index_vault_incrementally(str(target_path))

        print("\n Sync complete. Resuming watch...")

if __name__ == "__main__":
    start_watcher()