import os
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class EmbeddingManager:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.index_path = f"faiss_index/user_{user_id}"

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        self.vector_store = None

    def load_or_create(self):
        """
        Load existing FAISS or create new empty one
        """
        if os.path.exists(self.index_path):
            self.vector_store = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.vector_store = None

    def add_documents(self, documents: List[Document]):
        """
        Add documents to vector store
        """
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)

        os.makedirs(self.index_path, exist_ok=True)
        self.vector_store.save_local(self.index_path)

    def retrieve(self, query: str, k: int = 5):
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")

        return self.vector_store.similarity_search(query, k=k)