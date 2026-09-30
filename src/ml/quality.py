from typing import Tuple
import numpy as np
try:
    from sklearn.ensemble import IsolationForest
except ImportError:
    from sklearn.ensemble._iforest import IsolationForest


def detect_anomalies(
    embeddings: np.ndarray,
    contamination: float = 0.05
) -> np.ndarray:
    """
    Identifica documentos/chunks que são outliers em relação ao restante corpus. Retorna True para anomalias e False para pontos normais.
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    # -1 para anomalia, 1 para ponto normal
    preds = iso_forest.fit_predict(embeddings)
    return preds == -1