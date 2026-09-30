from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v1"):
        """Carrega o modelo de embeddings localmente."""
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str], show_progress_bar: bool = True) -> np.ndarray:
        """Gera vetores numéricos para uma lista de textos."""
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )
        return embeddings