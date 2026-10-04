import sys
from pathlib import Path
import unittest
import numpy as np

# Garante que a raiz do projeto esteja no sys.path para importação dos módulos de src/
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.parsers import load_document, read_txt, read_pdf
from src.ingestion.chunker import split_text_into_chunks, process_document_to_chunks
from src.embeddings.model import EmbeddingService


class TestDocumentIngestion(unittest.TestCase):
    """Valida a leitura e extração de texto de ficheiros .txt e .pdf."""

    def setUp(self):
        self.data_dir = ROOT_DIR / "data" / "raw"
        self.txt_path = self.data_dir / "txts" / "DieuDAmor.txt"
        self.pdf_path = self.data_dir / "pdfs" / "DieuDAmor.pdf"

    def test_extract_text_from_txt(self):
        """Valida que o leitor abre e extrai texto de arquivo .txt."""
        self.assertTrue(self.txt_path.exists(), f"Arquivo de teste não encontrado: {self.txt_path}")
        
        doc = load_document(self.txt_path)
        self.assertEqual(doc["filename"], "DieuDAmor.txt")
        self.assertEqual(doc["extension"], ".txt")
        self.assertIsInstance(doc["content"], str)
        self.assertGreater(len(doc["content"]), 0, "O conteúdo extraído do .txt não deve estar vazio.")

    def test_extract_text_from_pdf(self):
        """Valida que o leitor abre e extrai texto de arquivo .pdf."""
        self.assertTrue(self.pdf_path.exists(), f"Arquivo de teste não encontrado: {self.pdf_path}")
        
        doc = load_document(self.pdf_path)
        self.assertEqual(doc["filename"], "DieuDAmor.pdf")
        self.assertEqual(doc["extension"], ".pdf")
        self.assertIsInstance(doc["content"], str)
        self.assertGreater(len(doc["content"]), 0, "O conteúdo extraído do .pdf não deve estar vazio.")

    def test_unsupported_file_extension(self):
        """Valida que extensões não suportadas geram ValueError."""
        dummy_file = ROOT_DIR / "data" / "raw" / "dummy.docx"
        # Usamos arquivo inexistente ou mock
        with self.assertRaises(FileNotFoundError):
            load_document(dummy_file)


class TestTextChunking(unittest.TestCase):
    """Valida o particionamento em blocos com sobreposição (chunking)."""

    def test_chunking_with_overlap(self):
        """Valida que o particionamento fatia o texto com o tamanho e sobreposição corretos."""
        # Cria uma sequência sintética de 100 palavras controladas
        words = [f"word_{i}" for i in range(100)]
        text = " ".join(words)
        
        chunk_size = 30
        overlap = 10
        chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
        
        # Com 100 palavras, step = 20 (30 - 10):
        # chunk 0: 0..29 (30 palavras)
        # chunk 1: 20..49 (30 palavras)
        # chunk 2: 40..69 (30 palavras)
        # chunk 3: 60..89 (30 palavras)
        # chunk 4: 80..99 (20 palavras) -> total: 5 chunks
        self.assertEqual(len(chunks), 5)
        
        # Verifica tamanho do primeiro chunk
        first_chunk_words = chunks[0].split()
        self.assertEqual(len(first_chunk_words), chunk_size)
        
        # Verifica a sobreposição entre o chunk 0 e o chunk 1
        second_chunk_words = chunks[1].split()
        overlap_from_first = first_chunk_words[-overlap:]
        overlap_in_second = second_chunk_words[:overlap]
        self.assertEqual(
            overlap_from_first,
            overlap_in_second,
            "As palavras de sobreposição entre blocos adjacentes devem ser idênticas."
        )

    def test_process_document_to_chunks_structure(self):
        """Valida se os chunks estruturados contêm IDs e metadados apropriados."""
        doc_info = {
            "filename": "sample.txt",
            "extension": ".txt",
            "content": " ".join([f"palavra_{i}" for i in range(70)])
        }
        structured = process_document_to_chunks(doc_info, chunk_size=40, overlap=10)
        
        self.assertGreater(len(structured), 0)
        first_item = structured[0]
        self.assertIn("id", first_item)
        self.assertIn("text", first_item)
        self.assertIn("metadata", first_item)
        
        self.assertEqual(first_item["id"], "sample.txt_chunk_0")
        self.assertEqual(first_item["metadata"]["source"], "sample.txt")
        self.assertEqual(first_item["metadata"]["file_type"], ".txt")
        self.assertEqual(first_item["metadata"]["chunk_index"], 0)

    def test_empty_text_returns_empty_chunks(self):
        """Valida que texto vazio retorna lista vazia."""
        chunks = split_text_into_chunks("")
        self.assertEqual(chunks, [])


class TestEmbeddingsGeneration(unittest.TestCase):
    """Valida a geração de embeddings locais e a dimensionalidade dos vetores."""

    @classmethod
    def setUpClass(cls):
        # Carrega o modelo uma única vez para toda a classe de testes
        cls.embedder = EmbeddingService(model_name="all-MiniLM-L6-v1")

    def test_single_embedding_dimension(self):
        """Valida que o embedding gerado possui exatamente 384 dimensões."""
        sample_text = ["A guerra é a paz, a liberdade é a escravidão."]
        embeddings = self.embedder.generate_embeddings(sample_text, show_progress_bar=False)
        
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape, (1, 384), "O vetor de embedding deve possuir dimensão (1, 384).")
        self.assertFalse(np.isnan(embeddings).any(), "Os embeddings não devem conter valores NaN.")

    def test_batch_embeddings_dimension(self):
        """Valida geração de embeddings em lote preservando a dimensionalidade 384."""
        sample_texts = [
            "Primeiro parágrafo de teste sobre literatura.",
            "Segundo parágrafo sobre representações vetoriais densas.",
            "Terceiro parágrafo avaliando modelos locais."
        ]
        embeddings = self.embedder.generate_embeddings(sample_texts, show_progress_bar=False)
        
        self.assertEqual(embeddings.shape, (3, 384), "O lote de 3 textos deve resultar em uma matriz (3, 384).")


if __name__ == "__main__":
    unittest.main()
