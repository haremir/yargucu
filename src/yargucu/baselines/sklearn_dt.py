"""Scikit-learn Karar Ağacı temel model (baseline) modülü."""

import time
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier as SklearnDT

from yargucu.evaluation.metrics import evaluate_predictions


def get_sklearn_model(**kwargs: Any) -> SklearnDT:
    """Scikit-learn DecisionTreeClassifier model nesnesini döndürür."""
    params: dict[str, Any] = {
        "max_depth": 5,
        "criterion": "entropy",
        "class_weight": "balanced",
        "random_state": 42,
    }
    params.update(kwargs)
    return SklearnDT(**params)


def train_sklearn_dt(
    X_train: object,
    y_train: object,
    X_test: object,
    y_test: object,
    **kwargs: Any,
) -> dict[str, Any]:
    """Scikit-learn DecisionTreeClassifier modelini eğitir ve manuel metriklerle değerlendirir.

    Args:
        X_train: Eğitim öznitelik matrisi.
        y_train: Eğitim hedef etiketleri.
        X_test: Test öznitelik matrisi.
        y_test: Test hedef etiketleri.
        **kwargs: DecisionTreeClassifier modeline iletilecek hiperparametreler.

    Returns:
        dict[str, Any]: Model nesnesi, eğitim süresi ve test metriklerini içeren sözlük.
    """
    X_tr = np.asarray(X_train, dtype=float)
    y_tr = np.asarray(y_train, dtype=int)
    X_te = np.asarray(X_test, dtype=float)
    y_te = np.asarray(y_test, dtype=int)

    model = get_sklearn_model(**kwargs)

    start_time = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_time = time.perf_counter() - start_time

    train_preds = model.predict(X_tr)
    test_preds = model.predict(X_te)

    train_metrics = evaluate_predictions(y_tr, train_preds)
    test_metrics = evaluate_predictions(y_te, test_preds)

    return {
        "model_name": "Sklearn Decision Tree (Balanced)",
        "model": model,
        "train_time_sec": train_time,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
    }
