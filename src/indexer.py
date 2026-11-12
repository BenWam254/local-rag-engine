import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import duckdb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util, CrossEncoder

class HybridIndexer:
    def __init__(self, db_path: str = "vault_index.duckdb", model_name: str = "all-MiniLM-L6-v2", reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Initializing Persistent Hybrid Indexer (DuckDB: {db_path})...")
        self.db_path = db_path
        self.encoder = SentenceTransformer(model_name)
        
        print(f"Loading Cross-Encoder Reranker ({reranker_model})...")
        self.reranker = CrossEncoder(reranker_model)
        
        self.documents: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.document_embeddings = None
        
        self._init_db()

    def _init_db(self):
        """Initializes the local DuckDB persistence schema."""
        with duckdb.connect(self.db_path) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS file_registry (
                    file_path VARCHAR PRIMARY KEY,
                    file_hash VARCHAR,
                    last_modified DOUBLE
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    file_path VARCHAR,
                    header VARCHAR,
                    text VARCHAR,
                    embedding BLOB
                )
            """)

    def _compute_file_hash(self, file_path: Path) -> str:
        """Computes MD5 hash of a file to detect changes."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def index_vault_incrementally(self, vault_dir: str):
        """
        Scans the vault, detects added/modified files via hashes, 
        and updates only the necessary chunks in DuckDB.
        """
        vault_path = Path(vault_dir)
        if not vault_path.exists():
            print(f"Vault directory '{vault_dir}' does not exist.")
            return

        current_files = list(vault_path.glob("**/*.md"))
        current_file_paths = {str(f.resolve()) for f in current_files}

        with duckdb.connect(self.db_path) as con:
            # Get existing registered files from DB
            existing_registry = {
                row[0]: {"hash": row[1], "mtime": row[2]} 
                for row in con.execute("SELECT file_path, file_hash, last_modified FROM file_registry").fetchall()
            }

            # Identify deleted files
            deleted_files = set(existing_registry.keys()) - current_file_paths
            if deleted_files:
                print(f"Removing {len(deleted_files)} deleted files from index...")
                for f_path in deleted_files:
                    con.execute("DELETE FROM chunks WHERE file_path = ?", [f_path])
                    con.execute("DELETE FROM file_registry WHERE file_path = ?", [f_path])

            files_to_process = []
            for file_path in current_files:
                f_str = str(file_path.resolve())
                f_hash = self._compute_file_hash(file_path)
                f_mtime = file_path.stat().st_mtime

                if f_str not in existing_registry or existing_registry[f_str]["hash"] != f_hash:
                    files_to_process.append((file_path, f_str, f_hash, f_mtime))

            if files_to_process:
                print(f"Found {len(files_to_process)} new or modified files. Re-indexing...")
                
                for file_path, f_str, f_hash, f_mtime in files_to_process:
                    # Remove old chunks for this specific file if it was modified
                    con.execute("DELETE FROM chunks WHERE file_path = ?", [f_str])
                    
                    # Parse single file into semantic chunks by headers
                    content = file_path.read_text(encoding="utf-8")
                    file_name = file_path.name
                    file_chunks = []
                    
                    current_header = "Overview"
                    current_text_lines = []
                    
                    for line in content.splitlines():
                        if line.startswith("#"):
                            if current_text_lines:
                                file_chunks.append({
                                    "file_name": file_name,
                                    "file_path": f_str,
                                    "header": current_header,
                                    "text": "\n".join(current_text_lines).strip()
                                })
                                current_text_lines = []
                            current_header = line.lstrip("#").strip()
                        else:
                            current_text_lines.append(line)
                            
                    if current_text_lines:
                        file_chunks.append({
                            "file_name": file_name,
                            "file_path": f_str,
                            "header": current_header,
                            "text": "\n".join(current_text_lines).strip()
                        })

                    if file_chunks:
                        texts = [c["text"] for c in file_chunks]
                        embeddings = self.encoder.encode(texts, convert_to_tensor=False, normalize_embeddings=True)

                        for chunk, emb in zip(file_chunks, embeddings):
                            con.execute(
                                "INSERT INTO chunks (file_path, header, text, embedding) VALUES (?, ?, ?, ?)",
                                [f_str, chunk["header"], chunk["text"], emb.astype(np.float32).tobytes()]
                            )

                    # Update file registry
                    con.execute(
                        "INSERT OR REPLACE INTO file_registry (file_path, file_hash, last_modified) VALUES (?, ?, ?)",
                        [f_str, f_hash, f_mtime]
                    )
            else:
                print("No vault changes detected. Loaded index directly from DuckDB persistence!")

            # Load all active chunks from DB into memory structures for lightning-fast retrieval
            rows = con.execute("SELECT file_path, header, text, embedding FROM chunks").fetchall()
            self.documents = []
            embeddings_list = []
            
            for row in rows:
                self.documents.append({
                    "file_name": Path(row[0]).name,
                    "file_path": row[0],
                    "header": row[1],
                    "text": row[2]
                })
                emb_array = np.frombuffer(row[3], dtype=np.float32)
                embeddings_list.append(emb_array)

            if embeddings_list:
                self.document_embeddings = torch.tensor(np.array(embeddings_list))
                texts = [doc["text"] for doc in self.documents]
                self.corpus_tokens = [doc.lower().split() for doc in texts]
                self.bm25 = BM25Okapi(self.corpus_tokens)
            
            print(f"Index ready! Total active chunks in memory/DB: {len(self.documents)}")

    def search_dense(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if self.document_embeddings is None or not self.documents:
            return []
        query_embedding = self.encoder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        scores = util.dot_score(query_embedding, self.document_embeddings)[0].cpu().numpy()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def search_bm25(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if not self.bm25 or not self.documents:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def hybrid_search(self, query: str, top_k: int = 4, rerank: bool = True, k_constant: int = 60) -> List[Dict[str, Any]]:
        """
        Executes hybrid search (RRF) to get a broad candidate pool, 
        then reranks them using a Cross-Encoder for maximum precision.
        """
        fetch_k = top_k * 3  # Fetch a wider net of candidates first
        dense_results = self.search_dense(query, top_k=fetch_k)
        bm25_results = self.search_bm25(query, top_k=fetch_k)

        # 1. Base Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[int, float] = {}
        for rank, (doc_idx, _) in enumerate(dense_results, start=1):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k_constant + rank))
        for rank, (doc_idx, _) in enumerate(bm25_results, start=1):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k_constant + rank))

        # Sort by RRF to get our initial candidate list
        sorted_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)[:fetch_k]
        
        candidates = []
        for idx in sorted_indices:
            doc = self.documents[idx].copy()
            doc["base_score"] = rrf_scores[idx]
            candidates.append(doc)

        if not rerank or not candidates:
            return candidates[:top_k]

        # 2. Cross-Encoder Reranking
        print("Reranking candidates for precision...")
        cross_inp = [[query, doc["text"]] for doc in candidates]
        cross_scores = self.reranker.predict(cross_inp)

        for i, score in enumerate(cross_scores):
            candidates[i]["rerank_score"] = float(score)

        # Sort purely by the Cross-Encoder's surgical precision score
        reranked_results = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked_results[:top_k]