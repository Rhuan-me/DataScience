from pathlib import Path
from src.ingestion.parsers import load_document
from src.ingestion.chunker import process_document_to_chunks
from src.embeddings.model import EmbeddingService
from src.storage.vector_store import LocalVectorStore


def run_pipeline() -> None:
    data_dir = Path("data/raw")
    files = list(data_dir.glob("**/*.*"))
    supported_files = [f for f in files if f.suffix.lower() in [".txt", ".pdf"]]

    if not supported_files:
        print("Nenhum arquivo .txt ou .pdf encontrado em data/raw/!")
        return

    print(f"Arquivos encontrados: {len(supported_files)}")
    all_chunks = []

    for file_path in supported_files:
        print(f"Lendo: {file_path.name}...")
        doc = load_document(file_path)
        chunks = process_document_to_chunks(doc)
        all_chunks.extend(chunks)

    print(f"Total de chunks gerados: {len(all_chunks)}")

    print("Carregando modelo de embeddings...")
    embedder = EmbeddingService()
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.generate_embeddings(texts)

    print("Salvando no ChromaDB...")
    db = LocalVectorStore()
    db.add_chunks(all_chunks, embeddings.tolist())

    print("Ingestão concluída com sucesso!")


if __name__ == "__main__":
    run_pipeline()