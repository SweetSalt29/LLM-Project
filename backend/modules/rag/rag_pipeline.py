import os
import base64
from pathlib import Path

import requests

from backend.modules.rag.rag_loader import MultimodalLoader, prepare_documents
from backend.modules.rag.embeddings import EmbeddingManager


VISION_MODEL = "qwen/qwen3-vl-235b-thinking:free"
TEXT_MODEL   = "meta-llama/llama-3-8b-instruct"


class RAGPipeline:
    def __init__(self, user_id: int):
        self.user_id      = user_id
        self.loader       = MultimodalLoader()
        self.embedder     = EmbeddingManager(user_id)
        self.api_key      = os.getenv("OPENROUTER_API_KEY")
        self.text_model   = TEXT_MODEL
        self.vision_model = VISION_MODEL

    # ----------------------------------------------------------------
    # BASE LLM CALL
    # ----------------------------------------------------------------
    def _call_llm(self, messages: list, model: str) -> str:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type":  "application/json"
            },
            json={"model": model, "messages": messages},
            timeout=90
        )
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"LLM API error: {data['error']}")
        return data["choices"][0]["message"]["content"].strip()

    def call_llm(self, messages: list) -> str:
        """Text-only call — used for rewriting, summarization, plain answers."""
        return self._call_llm(messages, self.text_model)

    # ----------------------------------------------------------------
    # MULTIMODAL LLM CALL
    # Sends text context + actual image bytes to Qwen3-VL so it can
    # reason precisely over both at query time.
    # ----------------------------------------------------------------
    def call_vision_llm(
        self,
        query: str,
        text_context: str,
        image_docs: list,
        history: list
    ) -> str:
        """
        Build a multimodal OpenRouter message and call Qwen3-VL.

        Each image_doc carries:
          - page_content : Qwen3-VL caption generated at ingest time
          - metadata["image_path"] : path to image file on disk
        """
        # System block — text context first
        system_content = [
            {
                "type": "text",
                "text": (
                    "You are an intelligent document assistant with vision capabilities. "
                    "Answer questions using the provided document text AND the images shown. "
                    "When the answer involves an image, describe exactly what you see and reference it explicitly. "
                    "If the answer is not present in the context or images, say: "
                    "'I could not find this in the uploaded documents.' "
                    "Be concise, accurate, and conversational."
                    f"\n\nDocument text context:\n{text_context}"
                )
            }
        ]

        # Attach each retrieved image with its caption as a label
        for img_doc in image_docs:
            image_path = img_doc.metadata.get("image_path", "")
            if not image_path or not Path(image_path).exists():
                continue

            try:
                with open(image_path, "rb") as f:
                    image_bytes = f.read()

                ext = Path(image_path).suffix.lower().lstrip(".")
                media_type = {
                    "jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png",  "gif":  "image/gif",
                    "webp": "image/webp", "bmp": "image/bmp",
                }.get(ext, "image/png")

                b64     = base64.b64encode(image_bytes).decode("utf-8")
                caption = img_doc.page_content
                source  = img_doc.metadata.get("source", "document")
                page    = img_doc.metadata.get("page", "?")

                # Text label precedes the image so the model knows context
                system_content.append({
                    "type": "text",
                    "text": f"\n[Image from '{source}', page {page}]\nPre-generated description: {caption}\n"
                })
                system_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{b64}"}
                })

            except Exception as e:
                print(f"[RAGPipeline] Could not attach image {image_path}: {e}")

        # Full messages list: system → history → user question
        messages = [{"role": "user", "content": system_content}]

        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})

        return self._call_llm(messages, self.vision_model)

    # ----------------------------------------------------------------
    # QUERY REWRITING
    # ----------------------------------------------------------------
    def rewrite_query(self, query: str, history: list) -> str:
        if not history:
            return query

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
                    "- If the question is genuinely standalone, return it unchanged.\n"
                    "- Output ONLY the rewritten question — no explanation, no preamble.\n\n"
                    f"Conversation history:\n{history_text}\n\n"
                    f"Follow-up question: {query}\n\n"
                    "Rewritten standalone question:"
                )
            }
        ]

        try:
            rewritten = self.call_llm(messages).strip()
            if not rewritten or len(rewritten) > 500:
                return query
            return rewritten
        except Exception:
            return query

    # ----------------------------------------------------------------
    # MAIN QUERY METHOD
    # ----------------------------------------------------------------
    def query(self, query: str, history: list = None) -> dict:
        """
        Multi-turn RAG query with automatic multimodal escalation.

        Steps:
          1. Rewrite query to resolve pronouns / vague references
          2. Retrieve top-6 chunks from FAISS (text + image chunks mixed)
          3. Split into text_docs and image_docs
          4a. If image_docs present → Qwen3-VL with text context + images
          4b. Else → fast text-only LLM (cheaper, quicker)
          5. Return answer, sources, and a used_vision flag
        """
        if history is None:
            history = []

        self.embedder.load_or_create()

        # Step 1
        retrieval_query = self.rewrite_query(query, history)

        # Step 2
        docs = self.embedder.retrieve(retrieval_query, k=6)

        # Step 3
        text_docs  = [d for d in docs if not d.metadata.get("is_image_chunk")]
        image_docs = [d for d in docs if d.metadata.get("is_image_chunk")]

        text_context = "\n\n".join([d.page_content for d in text_docs])

        # Step 4
        if image_docs:
            # Vision path — image chunks were retrieved, use Qwen3-VL
            answer = self.call_vision_llm(
                query=query,
                text_context=text_context,
                image_docs=image_docs,
                history=history
            )
        else:
            # Text-only path
            system_prompt = {
                "role": "user",
                "content": (
                    "You are an intelligent document assistant. "
                    "Answer questions using ONLY the provided document context. "
                    "If the answer is not in the context, say: "
                    "'I could not find this in the uploaded documents.' "
                    "Be concise and conversational. Remember the conversation history."
                    f"\n\nDocument Context:\n{text_context}"
                )
            }
            messages = [system_prompt]
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": query})

            answer = self.call_llm(messages)

        # Step 5 — deduplicate sources
        seen = set()
        unique_sources = []
        for doc in docs:
            source = doc.metadata.get("source")
            page   = doc.metadata.get("page")
            key    = (source, page)
            if key not in seen:
                seen.add(key)
                unique_sources.append({
                    "source":   source,
                    "page":     page,
                    "is_image": doc.metadata.get("is_image_chunk", False)
                })

        return {
            "answer":          answer,
            "retrieval_query": retrieval_query,
            "sources":         unique_sources,
            "used_vision":     len(image_docs) > 0
        }

    # ----------------------------------------------------------------
    # SUMMARIZE CONVERSATION
    # ----------------------------------------------------------------
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
                    f"Conversation:\n{history_text}\n\nSummary:"
                )
            }
        ]

        return self.call_llm(messages)