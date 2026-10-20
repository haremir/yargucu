"""Model performans değerlendirme metrikleri (Doğruluk, F1, Hassasiyet, Duyarlılık vb.).

Bu modül scikit-learn kullanılmadan, tamamen NumPy ile ikili (binary) sınıflandırma
metriklerini hesaplamak üzere sıfırdan yazılmıştır.
"""

from typing import Any

import numpy as np


def confusion_matrix(
    y_true: object,
    y_pred: object,
) -> np.ndarray:
    """İkili sınıflandırma için karmaşıklık matrisini (Confusion Matrix) hesaplar.

    Matris Düzeni:
        [[TN, FP],
         [FN, TP]]

    Args:
        y_true: Gerçek sınıf etiketleri (0 veya 1).
        y_pred: Tahmin edilen sınıf etiketleri (0 veya 1).

    Returns:
        np.ndarray: (2, 2) boyutlu karmaşıklık matrisi.
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)

    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tp = int(np.sum((yt == 1) & (yp == 1)))

    return np.array([[tn, fp], [fn, tp]], dtype=int)


def accuracy_score(y_true: object, y_pred: object) -> float:
    """Doğruluk (Accuracy) skorunu hesaplar: (TP + TN) / Toplam Örnek.

    Args:
        y_true: Gerçek sınıf etiketleri.
        y_pred: Tahmin edilen sınıf etiketleri.

    Returns:
        float: [0.0, 1.0] aralığında doğruluk skoru.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    if yt.size == 0:
        return 0.0

    return float(np.mean(yt == yp))


def precision_score(y_true: object, y_pred: object, pos_label: int = 1) -> float:
    """Kesinlik / Hassasiyet (Precision) skorunu hesaplar: TP / (TP + FP).

    Args:
        y_true: Gerçek sınıf etiketleri.
        y_pred: Tahmin edilen sınıf etiketleri.
        pos_label: Pozitif sınıf etiketi (varsayılan: 1).

    Returns:
        float: [0.0, 1.0] aralığında kesinlik skoru. Payda 0 ise 0.0 döner.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    tp = np.sum((yt == pos_label) & (yp == pos_label))
    fp = np.sum((yt != pos_label) & (yp == pos_label))

    total_positive_predictions = tp + fp
    if total_positive_predictions == 0:
        return 0.0

    return float(tp / total_positive_predictions)


def recall_score(y_true: object, y_pred: object, pos_label: int = 1) -> float:
    """Duyarlılık / Anma (Recall) skorunu hesaplar: TP / (TP + FN).

    Args:
        y_true: Gerçek sınıf etiketleri.
        y_pred: Tahmin edilen sınıf etiketleri.
        pos_label: Pozitif sınıf etiketi (varsayılan: 1).

    Returns:
        float: [0.0, 1.0] aralığında duyarlılık skoru. Payda 0 ise 0.0 döner.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    tp = np.sum((yt == pos_label) & (yp == pos_label))
    fn = np.sum((yt == pos_label) & (yp != pos_label))

    actual_positives = tp + fn
    if actual_positives == 0:
        return 0.0

    return float(tp / actual_positives)


def f1_score(y_true: object, y_pred: object, pos_label: int = 1) -> float:
    """F1 skorunu hesaplar: 2 * (Precision * Recall) / (Precision + Recall).

    Args:
        y_true: Gerçek sınıf etiketleri.
        y_pred: Tahmin edilen sınıf etiketleri.
        pos_label: Pozitif sınıf etiketi (varsayılan: 1).

    Returns:
        float: [0.0, 1.0] aralığında F1 skoru.
    """
    prec = precision_score(y_true, y_pred, pos_label=pos_label)
    rec = recall_score(y_true, y_pred, pos_label=pos_label)

    if prec + rec == 0.0:
        return 0.0

    return float(2.0 * (prec * rec) / (prec + rec))


def evaluate_predictions(y_true: object, y_pred: object) -> dict[str, Any]:
    """Tüm metrikleri ve karmaşıklık matrisini tek seferde hesaplayıp sözlük olarak döndürür.

    Args:
        y_true: Gerçek sınıf etiketleri.
        y_pred: Tahmin edilen sınıf etiketleri.

    Returns:
        dict[str, Any]: accuracy, precision, recall, f1 ve confusion_matrix içeren sözlük.
    """
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }
