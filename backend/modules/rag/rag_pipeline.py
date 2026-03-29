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
    # Accepts a full messages list for multi-turn support
    # -----------------------------
    def call_llm(self, messages: list) -> str:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": messages
            }
        )
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    # -----------------------------
    # QUERY WITH CONVERSATION MEMORY
    # history: list of {"role": "user"/"assistant", "content": str}
    # -----------------------------
    def query(self, query: str, history: list = None) -> dict:
        if history is None:
            history = []

        self.embedder.load_or_create()
        docs = self.embedder.retrieve(query, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])

        # System prompt — sets the assistant's behaviour
        system_prompt = {
            "role": "user",
            "content": (
                "You are an intelligent document assistant. "
                "Answer questions using ONLY the provided document context. "
                "If the answer is not in the context, say: "
                "'I could not find this in the uploaded documents.' "
                "Be concise and conversational. Remember the conversation history."
                f"\n\nDocument Context:\n{context}"
            )
        }

        # Build messages: system → history → current question
        messages = [system_prompt]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": query})

        answer = self.call_llm(messages)

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

    # -----------------------------
    # SUMMARIZE CONVERSATION
    # -----------------------------
    def summarize_conversation(self, history: list) -> str:
        if not history:
            return "No conversation to summarize yet."

        history_text = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history
        ])

        messages = [
            {
                "role": "user",
                "content": (
                    "Summarize the following conversation between a user and a document assistant. "
                    "Highlight: key questions asked, main answers given, and important findings.\n\n"
                    f"Conversation:\n{history_text}\n\n"
                    "Summary:"
                )
            }
        ]

        return self.call_llm(messages)