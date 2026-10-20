"""XGBoost sınıflandırıcı temel model (baseline) modülü."""

import time
from typing import Any

import numpy as np
from xgboost import XGBClassifier

from yargucu.evaluation.metrics import evaluate_predictions


def get_xgboost_model(**kwargs: Any) -> XGBClassifier:
    """XGBoost XGBClassifier model nesnesini döndürür."""
    params: dict[str, Any] = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    params.update(kwargs)
    return XGBClassifier(**params)


def train_xgboost(
    X_train: object,
    y_train: object,
    X_test: object,
    y_test: object,
    **kwargs: Any,
) -> dict[str, Any]:
    """XGBoost modelini eğitir ve manuel metriklerle değerlendirir.

    Sınıf dengesizliğini dengelemek için varsayılan olarak scale_pos_weight
    oranı otomatik hesaplanıp modele aktarılır.

    Args:
        X_train: Eğitim öznitelik matrisi.
        y_train: Eğitim hedef etiketleri.
        X_test: Test öznitelik matrisi.
        y_test: Test hedef etiketleri.
        **kwargs: XGBClassifier modeline iletilecek hiperparametreler.

    Returns:
        dict[str, Any]: Model nesnesi, eğitim süresi ve test metriklerini içeren sözlük.
    """
    X_tr = np.asarray(X_train, dtype=float)
    y_tr = np.asarray(y_train, dtype=int)
    X_te = np.asarray(X_test, dtype=float)
    y_te = np.asarray(y_test, dtype=int)

    # Otomatik scale_pos_weight hesaplama (negatif / pozitif oranı)
    num_neg = float(np.sum(y_tr == 0))
    num_pos = float(np.sum(y_tr == 1))
    scale_pos = num_neg / max(1.0, num_pos)

    params: dict[str, Any] = {"scale_pos_weight": scale_pos}
    params.update(kwargs)

    model = get_xgboost_model(**params)

    start_time = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_time = time.perf_counter() - start_time

    train_preds = model.predict(X_tr)
    test_preds = model.predict(X_te)

    train_metrics = evaluate_predictions(y_tr, train_preds)
    test_metrics = evaluate_predictions(y_te, test_preds)

    return {
        "model_name": "XGBoost (Balanced)",
        "model": model,
        "train_time_sec": train_time,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
    }
