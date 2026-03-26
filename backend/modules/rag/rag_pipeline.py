from backend.modules.rag.rag_loader import MultimodalLoader, prepare_documents
from backend.modules.rag.embeddings import EmbeddingManager
import requests
import os


class RAGPipeline:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.loader = MultimodalLoader()
        self.embedder = EmbeddingManager(user_id)

        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = "meta-llama/llama-3-8b-instruct"

    # -----------------------------
    # LLM CALL (OpenRouter)
    # -----------------------------
    def call_llm(self, prompt: str):
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        )

        data = response.json()

        return data["choices"][0]["message"]["content"]

    # -----------------------------
    # QUERY (FULL RAG)
    # -----------------------------
    def query(self, query: str):
        self.embedder.load_or_create()
        docs = self.embedder.retrieve(query, k=5)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are an intelligent assistant.

Answer ONLY from the provided context.
If answer is not found, say:
"I could not find this in the uploaded documents."

Context:
{context}

Question:
{query}

Answer:
"""

        answer = self.call_llm(prompt)

        return {
            "answer": answer,
            "sources": [
                {
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page")
                }
                for doc in docs
            ]
        }