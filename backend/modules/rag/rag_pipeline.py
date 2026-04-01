import os
import base64
from pathlib import Path

import requests

from backend.modules.rag.rag_loader import MultimodalLoader, prepare_documents
from backend.modules.rag.embeddings import EmbeddingManager
from backend.modules.chat_memory import (
    get_standalone_context,
    save_standalone_message
)

REWRITE_MODEL = "meta-llama/llama-3.2-3b-instruct"
VISION_MODEL  = "qwen/qwen2.5-vl-7b-instruct"
TEXT_MODEL    = "meta-llama/llama-3-8b-instruct"


class RAGPipeline:
    def __init__(self, user_id: int):
        self.user_id       = user_id
        self.loader        = MultimodalLoader()
        self.embedder      = EmbeddingManager(user_id)
        self.api_key       = os.getenv("OPENROUTER_API_KEY")
        self.text_model    = TEXT_MODEL
        self.vision_model  = VISION_MODEL
        self.rewrite_model = REWRITE_MODEL

    # ----------------------------------------------------------------
    # LLM CALLS
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
        return self._call_llm(messages, self.text_model)

    def _call_rewrite_llm(self, messages: list) -> str:
        return self._call_llm(messages, self.rewrite_model)

    # ----------------------------------------------------------------
    # MULTIMODAL LLM CALL
    # ----------------------------------------------------------------
    def call_vision_llm(
        self,
        query: str,
        text_context: str,
        image_docs: list,
        context_str: str,
        doc_list_str: str
    ) -> str:
        system_content = [
            {
                "type": "text",
                "text": (
                    "You are an intelligent document assistant with vision capabilities.\n"
                    f"You have access to the following document(s): {doc_list_str}\n\n"
                    "When answering:\n"
                    "- Always attribute each fact or point to its source document by name.\n"
                    "- If multiple documents are relevant, list findings per document.\n"
                    "- If the answer is not found in any document, say: "
                    "'I could not find this in the uploaded documents.'\n"
                    "- Be concise, accurate, and conversational.\n\n"
                    f"Prior conversation context:\n{context_str}\n\n"
                    f"Document text context:\n{text_context}"
                )
            }
        ]

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
                system_content.append({
                    "type": "text",
                    "text": f"\n[Image from '{source}', page {page}]\nDescription: {caption}\n"
                })
                system_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{b64}"}
                })
            except Exception as e:
                print(f"[RAGPipeline] Could not attach image {image_path}: {e}")

        messages = [
            {"role": "user", "content": system_content},
            {"role": "user", "content": query}
        ]
        return self._call_llm(messages, self.vision_model)

    # ----------------------------------------------------------------
    # STANDALONE REWRITE (lightweight LLM)
    # ----------------------------------------------------------------
    def _rewrite_as_standalone(self, user_query: str, context_str: str) -> str:
        if not context_str.strip():
            return user_query

        messages = [
            {
                "role": "user",
                "content": (
                    "You are a query rewriting assistant.\n\n"
                    "Given prior conversation context and a follow-up question, "
                    "rewrite the follow-up into a fully self-contained, explicit question "
                    "that can be understood WITHOUT any prior context.\n\n"
                    "Rules:\n"
                    "- Replace ALL pronouns (it, they, this, that, he, she, its, their) "
                    "with the explicit noun they refer to from the context.\n"
                    "- Expand vague references like 'the document', 'the same', "
                    "'that topic', 'the above' to their specific subject.\n"
                    "- Never leave any pronoun or vague reference unresolved.\n"
                    "- If the question is already standalone, return it unchanged.\n"
                    "- Output ONLY the rewritten question — no explanation, no preamble.\n\n"
                    f"Prior conversation context:\n{context_str}\n\n"
                    f"Follow-up question: {user_query}\n\n"
                    "Rewritten standalone question:"
                )
            }
        ]
        try:
            rewritten = self._call_rewrite_llm(messages).strip()
            if not rewritten or len(rewritten) > 500:
                return user_query
            return rewritten
        except Exception:
            return user_query

    # ----------------------------------------------------------------
    # ANSWER SUMMARIZER (lightweight LLM)
    # ----------------------------------------------------------------
    def _summarize_answer(self, standalone_query: str, answer: str) -> str:
        messages = [
            {
                "role": "user",
                "content": (
                    "Summarize the following answer in 2-3 sentences only. "
                    "Be concise and preserve the key facts.\n\n"
                    f"Question: {standalone_query}\n"
                    f"Answer: {answer}\n\n"
                    "2-3 sentence summary:"
                )
            }
        ]
        try:
            summary = self._call_rewrite_llm(messages).strip()
            return summary if summary else answer[:300]
        except Exception:
            return answer[:300]

    # ----------------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------------
    def _build_context_str(self, context_rows: list) -> str:
        if not context_rows:
            return ""
        lines = []
        for i, row in enumerate(context_rows, 1):
            lines.append(f"Turn {i}:")
            lines.append(f"  Q: {row['standalone_query']}")
            lines.append(f"  A: {row['answer_summary']}")
        return "\n".join(lines)

    def _build_doc_list_str(self, file_paths: list) -> str:
        """Human-readable list of filenames for the LLM system prompt."""
        names = [Path(p).name for p in file_paths]
        if not names:
            return "unknown document(s)"
        if len(names) == 1:
            return names[0]
        return ", ".join(names[:-1]) + f", and {names[-1]}"

    # ----------------------------------------------------------------
    # MAIN QUERY
    # ----------------------------------------------------------------
    def query(
        self,
        user_query: str,
        session_id: str,
        file_paths: list,
        mode: str = "rag"
    ) -> dict:
        """
        Multi-turn RAG query scoped to session-locked file_paths.

        Steps:
          1. Fetch standalone context (last 5 turns)
          2. Rewrite query to standalone
          3. Retrieve top-6 chunks filtered to session file_paths
          4. Generate answer (vision or text path)
          5. Summarize answer
          6. Save standalone turn
          7. Return answer + sources
        """
        self.embedder.load_or_create()

        # Step 1
        context_rows = get_standalone_context(session_id, mode, limit=5)
        context_str  = self._build_context_str(context_rows)
        doc_list_str = self._build_doc_list_str(file_paths)

        # Step 2
        standalone_query = self._rewrite_as_standalone(user_query, context_str)

        # Step 3 — filtered retrieval
        docs = self.embedder.retrieve(standalone_query, k=6, file_paths=file_paths)

        text_docs  = [d for d in docs if not d.metadata.get("is_image_chunk")]
        image_docs = [d for d in docs if d.metadata.get("is_image_chunk")]
        text_context = "\n\n".join([d.page_content for d in text_docs])

        # Step 4 — generate answer
        if image_docs:
            answer = self.call_vision_llm(
                query=user_query,
                text_context=text_context,
                image_docs=image_docs,
                context_str=context_str,
                doc_list_str=doc_list_str
            )
        else:
            # Build multi-doc attribution prompt
            multi_doc_instruction = (
                f"You have access to the following document(s): {doc_list_str}\n"
                "When answering:\n"
                "- Always attribute each fact or point to its source document by name.\n"
                "- If the question spans multiple documents, list findings per document "
                "with the source name clearly stated after each point.\n"
                "- If the answer is not found in any document, say: "
                "'I could not find this in the uploaded documents.'\n"
                "- Be concise and conversational."
            )

            system_prompt = {
                "role": "user",
                "content": (
                    f"You are an intelligent document assistant.\n"
                    f"{multi_doc_instruction}\n\n"
                    f"Prior conversation context:\n{context_str}\n\n"
                    f"Document Context:\n{text_context}"
                )
            }
            messages = [system_prompt, {"role": "user", "content": user_query}]
            answer = self.call_llm(messages)

        # Step 5
        answer_summary = self._summarize_answer(standalone_query, answer)

        # Step 6
        save_standalone_message(
            session_id=session_id,
            mode=mode,
            user_query=user_query,
            standalone_query=standalone_query,
            llm_answer=answer,
            answer_summary=answer_summary
        )

        # Step 7 — deduplicate sources
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
            "retrieval_query": standalone_query,
            "sources":         unique_sources,
            "used_vision":     len(image_docs) > 0
        }

    # ----------------------------------------------------------------
    # SUMMARIZE CONVERSATION
    # ----------------------------------------------------------------
    def summarize_conversation(self, session_id: str, mode: str = "rag") -> str:
        context_rows = get_standalone_context(session_id, mode, limit=5)
        if not context_rows:
            return "No conversation to summarize yet."

        context_str = self._build_context_str(context_rows)
        messages = [
            {
                "role": "user",
                "content": (
                    "Summarize the following document Q&A conversation. "
                    "Highlight: key questions asked, main answers given, and important findings.\n\n"
                    f"Conversation:\n{context_str}\n\nSummary:"
                )
            }
        ]
        return self.call_llm(messages)