from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.api.models.Collection import Collection


class LocalVectorStore:
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "knowledge_base"):
        self.persist_directory = persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection: Collection = self.client.get_or_create_collection(name=collection_name)

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """Insere ou atualiza chunks e seus respectivos embeddings no banco."""
        ids = [c["id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

    def search(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        """Realiza busca por similaridade de cosseno."""
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

    def get_all_vectors(self) -> Dict[str, Any]:
        """Retorna todos os embenddings, documentos e metadados do banco."""
        return self.collection.get(include=['embeddings', 'documents', 'metadatas'])