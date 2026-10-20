"""Gini indeksi (Gini Impurity) hesaplama modülü."""

import numpy as np


def calculate_gini(
    y: object,
    sample_weight: object | None = None,
) -> float:
    """Verilen etiket dağılımının Gini safsızlığını hesaplar.

    Formül:
        GI(S) = 1 - sum( p_i^2 )

    Args:
        y: Sınıf etiketleri (numpy array, liste veya pandas Series). Çok sınıflı
            (multi-class) etiketleri de destekler.
        sample_weight: Örnek ağırlıkları dizisi (opsiyonel, sınıf dengeleme için).

    Returns:
        float: Hesaplanmış Gini safsızlık değeri ([0, 1) aralığında). Boş veri için 0.0 döner.
    """
    labels = np.asarray(y)
    if labels.size == 0:
        return 0.0

    if sample_weight is None:
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / labels.size
    else:
        weights = np.asarray(sample_weight, dtype=float)
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0:
            return 0.0
        unique_labels = np.unique(labels)
        class_weights = np.array([np.sum(weights[labels == c]) for c in unique_labels], dtype=float)
        probabilities = class_weights / total_weight

    gini = 1.0 - np.sum(probabilities**2)

    if abs(gini) < 1e-12:
        return 0.0

    return float(gini)
