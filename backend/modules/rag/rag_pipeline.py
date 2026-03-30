from backend.modules.rag.rag_loader import MultimodalLoader, prepare_documents
from backend.modules.rag.embeddings import EmbeddingManager
import requests
import os


class RAGPipeline:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.loader  = MultimodalLoader()
        self.embedder = EmbeddingManager(user_id)

        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model   = "meta-llama/llama-3-8b-instruct"

    # -----------------------------
    # LLM CALL
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
    # QUERY REWRITING
    # Expands vague/pronoun-heavy queries into fully explicit ones
    # using recent conversation history before hitting the vector store.
    #
    # Example:
    #   History : User asked about "company_report.pdf"
    #   Query   : "what does it say about revenue?"
    #   Rewritten: "What does company_report.pdf say about revenue?"
    #
    # If the query is already self-contained, the LLM returns it unchanged.
    # -----------------------------
    def rewrite_query(self, query: str, history: list) -> str:
        """
        Rewrite the query into a fully standalone question using
        conversation history. Returns the rewritten query string.
        Falls back to the original query if rewriting fails.
        """
        # No history = first message, nothing to resolve
        if not history:
            return query

        # Only use the last 6 messages (3 turns) — enough context, not too noisy
        recent = history[-6:]
        history_text = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in recent
        ])

        messages = [
            {
                "role": "user",
                "content": (
                    "You are a query rewriting assistant.\n\n"
                    "Given the conversation history and a follow-up question, "
                    "rewrite the follow-up question into a fully self-contained, "
                    "explicit question that can be understood WITHOUT any prior context.\n\n"
                    "Rules:\n"
                    "- ALWAYS replace pronouns (it, they, this, that, he, she, its, their) "
                    "with the explicit noun they refer to from the conversation history.\n"
                    "- ALWAYS expand vague references like 'the document', 'the file', "
                    "'the same', 'that topic', 'the above' to their specific subject.\n"
                    "- NEVER leave any pronoun or vague reference unresolved.\n"
                    "- NEVER return the question unchanged if it contains any pronoun or reference.\n"
                    "- If the question is genuinely standalone with no pronouns or references, "
                    "return it unchanged.\n"
                    "- Output ONLY the rewritten question — no explanation, no preamble.\n\n"
                    f"Conversation history:\n{history_text}\n\n"
                    f"Follow-up question: {query}\n\n"
                    "Rewritten standalone question:"
                )
            }
        ]

        try:
            rewritten = self.call_llm(messages).strip()
            # Safety: if LLM returns something empty or way too long, fall back
            if not rewritten or len(rewritten) > 500:
                return query
            return rewritten
        except Exception:
            return query  # always fall back gracefully

    # -----------------------------
    # QUERY WITH CONVERSATION MEMORY
    # history: list of {"role": "user"/"assistant", "content": str}
    # -----------------------------
    def query(self, query: str, history: list = None) -> dict:
        if history is None:
            history = []

        self.embedder.load_or_create()

        # Step 1: Rewrite the query into a standalone explicit question
        # before hitting FAISS. Resolves pronouns and vague references.
        # If history is empty (first message), original query is returned unchanged.
        retrieval_query = self.rewrite_query(query, history)

        # Step 2: Retrieve using the rewritten query
        docs = self.embedder.retrieve(retrieval_query, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Step 3: Answer using original query + history + retrieved context
        # We pass the original query to the LLM (not rewritten) so the
        # response feels natural, not robotic
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

        messages = [system_prompt]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": query})

        answer = self.call_llm(messages)

        # Deduplicate sources — same file+page combo counts once
        # Filter out None pages since they add no useful info when duplicated
        seen = set()
        unique_sources = []
        for doc in docs:
            source = doc.metadata.get("source")
            page   = doc.metadata.get("page")
            key    = (source, page)
            if key not in seen:
                seen.add(key)
                unique_sources.append({"source": source, "page": page})

        return {
            "answer":           answer,
            "retrieval_query":  retrieval_query,
            "sources":          unique_sources
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