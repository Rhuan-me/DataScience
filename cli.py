import argparse
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


def run_search(query: str, n_results: int = 3) -> None:
    print(f"Buscando por: '{query}'...")
    embedder = EmbeddingService()
    db = LocalVectorStore()

    # Gera o vetor apenas da pergunta
    query_vec = embedder.generate_embeddings([query])[0].tolist()
    results = db.search(query_vec, n_results=n_results)

    if not results or not results["documents"] or not results["documents"][0]:
        print("Nenhum resultado encontrado.")
        return

    for idx, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"\n--- Resultado {idx + 1} ({meta['source']}) ---")
        print(doc[:300] + "...")


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI de Ingestão e Busca de Documentos")
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis")

    # Comando para rodar a ingestão
    subparsers.add_parser("ingest", help="Processa e ingere arquivos de data/raw/")

    # Comando para fazer busca semântica
    search_parser = subparsers.add_parser("search", help="Busca semântica no banco vetorial")
    search_parser.add_argument("query", type=str, help="Texto ou pergunta para buscar")
    search_parser.add_argument("-n", "--num", type=int, default=3, help="Número de resultados")

    args = parser.parse_args()

    if args.command == "ingest":
        run_pipeline()
    elif args.command == "search":
        run_search(args.query, args.num)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
