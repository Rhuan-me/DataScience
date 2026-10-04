from typing import Dict, Any, Tuple
import numpy as np
from sklearn.manifold import TSNE  # type: ignore[import-untyped]
from sklearn.cluster import KMeans  # type: ignore[import-untyped]


def run_clustering_and_tsne(
    embeddings: np.ndarray, n_clusters: int = 4, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Executa K-Means para agrupar os documentos e t-SNE para reduzir a dimensionalidade para 2D, permitindo a visualização.
    """
    # Agrupamento com K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    cluster_labels = kmeans.fit_predict(embeddings)

    # Redução de dimensionalidade para visualização 2D e perplexity ajustada para o número de amostras
    perplexity_val = min(30, max(5, len(embeddings) - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity_val,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    tsne_coords = tsne.fit_transform(embeddings)

    return cluster_labels, tsne_coords
