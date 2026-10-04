import argparse
from pathlib import Path
from src.ingestion.parsers import load_document
from src.ingestion.chunker import process_document_to_chunks
from src.embeddings.model import EmbeddingService
from src.storage.vector_store import LocalVectorStore


def run_pipeline(data_dir_path: str = "data/raw") -> None:
    data_dir = Path(data_dir_path)
    if not data_dir.exists():
        print(f"Diretório não encontrado: {data_dir_path}")
        return

    files = list(data_dir.glob("**/*.*"))
    supported_files = [f for f in files if f.suffix.lower() in [".txt", ".pdf"]]

    if not supported_files:
        print(f"Nenhum arquivo .txt ou .pdf encontrado em {data_dir_path}!")
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

    if not results or not results.get("documents") or not results["documents"][0]:
        print("Nenhum resultado encontrado.")
        return

    docs = results["documents"][0]
    metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
    dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

    for idx, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        source = meta.get("source", "desconhecido")
        chunk_idx = meta.get("chunk_index", "?")
        # ChromaDB utiliza squared L2 por padrão. Para vetores normalizados: cos_sim = 1 - (dist / 2)
        sim_score = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
        print(
            f"\n--- Resultado {idx + 1} | Fonte: {source} (Bloco #{chunk_idx}) | Distância: {dist:.4f} (Similaridade Cosseno: {sim_score:.2%}) ---"
        )
        # Normaliza espaços em branco repetidos para exibição limpa
        clean_text = " ".join(doc.split())
        print(clean_text[:300] + "...")


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI de Ingestão e Busca de Documentos")
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis")

    # Comando para rodar a ingestão
    ingest_parser = subparsers.add_parser(
        "ingest", help="Processa e ingere arquivos de documentos brutos"
    )
    ingest_parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Caminho para o diretório de dados brutos (padrão: data/raw)",
    )

    # Comando para fazer busca semântica
    search_parser = subparsers.add_parser("search", help="Busca semântica no banco vetorial")
    search_parser.add_argument("query", type=str, help="Texto ou pergunta para buscar")
    search_parser.add_argument(
        "-n",
        "--num",
        "-k",
        "--top-k",
        dest="num",
        type=int,
        default=3,
        help="Número de resultados a retornar (Top-K, padrão: 3)",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        run_pipeline(args.data_dir)
    elif args.command == "search":
        run_search(args.query, args.num)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
