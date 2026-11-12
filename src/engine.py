from typing import List, Dict, Any
import ollama

class RAGEngine:
    def __init__(self, indexer: Any, model_name: str = "llama3"):
        self.indexer = indexer
        self.model_name = model_name

    def query(self, prompt: str, top_k: int = 4) -> Dict[str, Any]:
        """
        Executes hybrid search to retrieve context and passes it to Ollama 
        for local offline synthesis.
        """
        print(f"Running hybrid search for query: '{prompt}'...")
        retrieved_chunks = self.indexer.hybrid_search(prompt, top_k=top_k)

        if not retrieved_chunks:
            return {
                "response": "No relevant context found in your local files.",
                "sources": []
            }

        # Construct context blocks with clean source attributions
        context_blocks = []
        sources = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            source_str = f"{chunk['file_name']} -> {chunk['header']}"
            sources.append(source_str)
            context_blocks.append(f"[Source {idx} ({source_str})]:\n{chunk['text']}")

        context_str = "\n\n".join(context_blocks)

        system_prompt = (
            "You are an offline local knowledge assistant. Answer the user's question accurately "
            "using *only* the provided context notes below. If the answer cannot be found in the context, "
            "state clearly that you don't know based on local files. Always cite your sources using "
            "the source labels (e.g., [Source 1]).\n\n"
            f"--- CONTEXT ---\n{context_str}\n--- END CONTEXT ---"
        )

        print(f"Generating answer locally using Ollama ({self.model_name})...")
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        answer = response["message"]["content"]
        return {
            "response": answer,
            "sources": list(set(sources))
        }
        