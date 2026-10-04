from typing import List, Dict, Any


def split_text_into_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Divide um texto longo em blocos de palavras com sobreposição (overlap)
    para manter a coerência semântica entre limites de chunks.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
    return chunks


def process_document_to_chunks(
    doc_info: Dict[str, Any], chunk_size: int = 300, overlap: int = 50
) -> List[Dict[str, Any]]:
    """Gera uma lista de chunks com metadados estruturados prontos para o ChromaDB."""
    raw_chunks = split_text_into_chunks(doc_info["content"], chunk_size, overlap)
    structured_chunks = []

    for idx, chunk in enumerate(raw_chunks):
        structured_chunks.append(
            {
                "id": f"{doc_info['filename']}_chunk_{idx}",
                "text": chunk,
                "metadata": {
                    "source": doc_info["filename"],
                    "file_type": doc_info["extension"],
                    "chunk_index": idx,
                },
            }
        )
    return structured_chunks
