"""Entropi hesaplama modülü."""

import numpy as np


def calculate_entropy(
    y: object,
    sample_weight: object | None = None,
) -> float:
    """Verilen etiket dağılımının Shannon entropisini hesaplar.

    Formül:
        H(S) = -sum( p_i * log2(p_i) )

    Args:
        y: Sınıf etiketleri (numpy array, liste veya pandas Series). Çok sınıflı
            (multi-class) etiketleri de destekler.
        sample_weight: Örnek ağırlıkları dizisi (opsiyonel, sınıf dengeleme için).

    Returns:
        float: Hesaplanmış entropi değeri (bit cinsinden). Boş veri için 0.0 döner.
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

    # p_i = 0 durumunda lim (p -> 0) p * log2(p) = 0 olduğundan sadece p > 0 ele alınır
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))

    if abs(entropy) < 1e-12:
        return 0.0

    return float(entropy)
