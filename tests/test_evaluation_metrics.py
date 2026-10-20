"""Değerlendirme metrikleri birim testleri."""

import numpy as np
import pytest

from yargucu.evaluation.metrics import (
    accuracy_score,
    confusion_matrix,
    evaluate_predictions,
    f1_score,
    precision_score,
    recall_score,
)


def test_confusion_matrix_binary() -> None:
    """İkili karmaşıklık matrisi hesaplama testi."""
    y_true = [1, 0, 1, 1, 0, 1, 0, 0]
    y_pred = [1, 0, 0, 1, 0, 1, 1, 0]

    cm = confusion_matrix(y_true, y_pred)
    # TN=3, FP=1, FN=1, TP=3
    expected_cm = np.array([[3, 1], [1, 3]])
    assert np.array_equal(cm, expected_cm)


def test_metrics_values() -> None:
    """Accuracy, Precision, Recall ve F1 skorlarının doğruluğunu test eder."""
    y_true = [1, 0, 1, 1, 0, 1, 0, 0]
    y_pred = [1, 0, 0, 1, 0, 1, 1, 0]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    assert pytest.approx(acc) == 6 / 8
    assert pytest.approx(prec) == 3 / 4
    assert pytest.approx(rec) == 3 / 4
    assert pytest.approx(f1) == 0.75


def test_perfect_predictions() -> None:
    """Mükemmel tahminlerde tüm skorlar 1.0 olmalıdır."""
    y_true = [0, 1, 0, 1, 1]
    y_pred = [0, 1, 0, 1, 1]

    metrics = evaluate_predictions(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_zero_division_safety() -> None:
    """Sıfıra bölünme durumlarında metrikler 0.0 dönmelidir."""
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]

    # Pozitif tahmin yok -> precision 0.0
    assert precision_score(y_true, y_pred) == 0.0
    # Pozitif gerçek yok -> recall 0.0
    assert recall_score(y_true, y_pred) == 0.0
    assert f1_score(y_true, y_pred) == 0.0
