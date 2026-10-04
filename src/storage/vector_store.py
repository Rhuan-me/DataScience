from pathlib import Path
from typing import List, Dict, Any, cast
import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection


class LocalVectorStore:
    def __init__(
        self, persist_directory: str = "./chroma_db", collection_name: str = "knowledge_base"
    ):
        self.persist_directory = persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_directory, settings=Settings(anonymized_telemetry=False)
        )
        self.collection: Collection = self.client.get_or_create_collection(name=collection_name)

    def add_chunks(
        self, chunks: List[Dict[str, Any]], embeddings: List[Any], batch_size: int = 500
    ) -> None:
        """Insere ou atualiza chunks e seus respectivos embeddings no banco em lotes."""
        ids = [c["id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        # Processamento em lotes com upsert (atualiza se já existir)
        for i in range(0, len(ids), batch_size):
            end_i = i + batch_size
            self.collection.upsert(
                ids=ids[i:end_i],
                documents=documents[i:end_i],
                metadatas=metadatas[i:end_i],
                embeddings=cast(Any, embeddings[i:end_i]),
            )

    def search(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        """Realiza busca por similaridade de cosseno."""
        results = self.collection.query(
            query_embeddings=cast(Any, [query_embedding]), n_results=n_results
        )
        return cast(Dict[str, Any], results)

    def get_all_vectors(self) -> Dict[str, Any]:
        """Retorna todos os embenddings, documentos e metadados do banco."""
        results = self.collection.get(include=["embeddings", "documents", "metadatas"])
        return cast(Dict[str, Any], results)
