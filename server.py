from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from src.indexer import HybridIndexer
from src.engine import RAGEngine

# Global dictionaary to hold our engine in memory
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Booting up Local RAG Engine into memory...")
    indexer = HybridIndexer(db_path="vault_index.duckdb", model_name="all-MiniLM-L6-v2")

    # Run a quick sync on startup, but trust the watcher for live updates
    indexer.index_vault_incrementally("vault_samples")
    ml_models["rag_engine"] = RAGEngine(indexer=indexer, model_name="llama3")

    print("API ready to receive queries.")
    yield
    # Clean up on shutdown
    ml_models.clear()

app = FastAPI(title="Local RAG API", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

@app.post("/api/chat")
async def chat(request: QueryRequest):
    print(f"\n Received query: {request.query}")

    # Execute the query using the hot engine in memory
    result = ml_models["rag_engine"].query(request.query)

    return{
        "query": request.query,
        "response": result["response"],
        "sources": result["sources"]
    }