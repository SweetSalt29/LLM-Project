import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class EmbeddingManager:
    def __init__(self, user_id: int):
        self.user_id    = user_id
        self.index_path = f"backend/modules/rag/faiss_index/user_{user_id}"

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        self.vector_store = None

    def load_or_create(self):
        """Load existing FAISS index or leave vector_store as None."""
        if os.path.exists(self.index_path):
            self.vector_store = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.vector_store = None

    def add_documents(self, documents: List[Document]):
        """Add documents to the user's FAISS index and persist."""
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)

        os.makedirs(self.index_path, exist_ok=True)
        self.vector_store.save_local(self.index_path)

    def retrieve(
        self,
        query: str,
        k: int = 6,
        file_paths: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Retrieve top-k chunks from FAISS.

        If file_paths is provided, only chunks whose metadata['file_path']
        matches one of the given paths are returned. This isolates each
        chat session to its locked knowledge base without needing separate
        FAISS indexes per file.

        Strategy: fetch k * 4 candidates then filter down to k so we
        always return enough results even after filtering.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. No documents have been embedded yet.")

        if not file_paths:
            # No filter — return global results (fallback only)
            return self.vector_store.similarity_search(query, k=k)

        # Normalise paths for comparison
        normalised = set(os.path.normpath(p) for p in file_paths)

        # Fetch a larger candidate pool to survive post-filter reduction
        candidates = self.vector_store.similarity_search(query, k=k * 4)

        filtered = [
            doc for doc in candidates
            if os.path.normpath(doc.metadata.get("file_path", "")) in normalised
        ]

        return filtered[:k]