"""Manuel karar ağacı eğitimi pipeline adımı."""

import sys
import time
from pathlib import Path

import pandas as pd

from yargucu.data.preprocess import run as run_preprocess
from yargucu.evaluation.metrics import evaluate_predictions
from yargucu.tree.decision_tree import DecisionTreeClassifier


def run() -> pd.DataFrame:
    """Sıfırdan yazılan Karar Ağacı modelini eğitir, değerlendirir ve kaydeder."""
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    print("=" * 65)
    print("MANUEL KARAR AĞACI MODELİ EĞİTİMİ (SCRATCH DECISION TREE)")
    print("=" * 65)

    train_path = Path("data/processed/train.csv")
    test_path = Path("data/processed/test.csv")

    if not train_path.exists() or not test_path.exists():
        print("İşlenmiş veri dosyaları bulunamadı, ön işleme adımı çalıştırılıyor...")
        run_preprocess()

    print("\n1. Veriler yükleniyor...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["income"]).values
    y_train = train_df["income"].values
    X_test = test_df.drop(columns=["income"]).values
    y_test = test_df["income"].values

    print(f"   - Eğitim Seti: {X_train.shape[0]:,} örnek, {X_train.shape[1]} öznitelik")
    print(f"   - Test Seti  : {X_test.shape[0]:,} örnek, {X_test.shape[1]} öznitelik")

    # Hiperparametreler
    max_depth = 5
    criterion = "entropy"
    class_weight = "balanced"
    print(f"\n2. Manuel Karar Ağacı eğitiliyor (max_depth={max_depth}, criterion='{criterion}', class_weight='{class_weight}')...")

    model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, class_weight=class_weight)

    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time
    print(f"   - Eğitim süresi: {train_time:.2f} saniye")

    print("\n3. Tahminler üretiliyor ve metrikler hesaplanıyor...")
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_metrics = evaluate_predictions(y_train, train_preds)
    test_metrics = evaluate_predictions(y_test, test_preds)

    print("\n4. Model Performans Özeti:")
    print(f"   - [Eğitim] Accuracy : %{train_metrics['accuracy'] * 100:.2f} | F1: {train_metrics['f1']:.4f}")
    print(f"   - [Test]   Accuracy : %{test_metrics['accuracy'] * 100:.2f} | F1: {test_metrics['f1']:.4f}")
    print(f"   - [Test]   Precision: %{test_metrics['precision'] * 100:.2f} | Recall: %{test_metrics['recall'] * 100:.2f}")

    results_data = [
        {
            "model": "Manual Decision Tree",
            "split": "train",
            "accuracy": round(train_metrics["accuracy"], 4),
            "precision": round(train_metrics["precision"], 4),
            "recall": round(train_metrics["recall"], 4),
            "f1": round(train_metrics["f1"], 4),
            "train_time_sec": round(train_time, 4),
        },
        {
            "model": "Manual Decision Tree",
            "split": "test",
            "accuracy": round(test_metrics["accuracy"], 4),
            "precision": round(test_metrics["precision"], 4),
            "recall": round(test_metrics["recall"], 4),
            "f1": round(test_metrics["f1"], 4),
            "train_time_sec": round(train_time, 4),
        },
    ]
    results_df = pd.DataFrame(results_data)

    # 5. Çıktıları kaydetme
    tables_dir = Path("outputs/tables")
    logs_dir = Path("outputs/logs")
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "manual_tree_results.csv"
    results_df.to_csv(csv_path, index=False)

    log_path = logs_dir / "manual_tree.log"
    log_content = (
        f"=== MANUEL KARAR AĞACI EĞİTİM LOGU ===\n"
        f"Tarih/Zaman: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Hiperparametreler: max_depth={max_depth}, criterion={criterion}\n"
        f"Eğitim Süresi: {train_time:.4f} saniye\n\n"
        f"EĞİTİM METRİKLERİ:\n"
        f"  - Accuracy : {train_metrics['accuracy']:.4f}\n"
        f"  - Precision: {train_metrics['precision']:.4f}\n"
        f"  - Recall   : {train_metrics['recall']:.4f}\n"
        f"  - F1-Score : {train_metrics['f1']:.4f}\n\n"
        f"TEST METRİKLERİ:\n"
        f"  - Accuracy : {test_metrics['accuracy']:.4f}\n"
        f"  - Precision: {test_metrics['precision']:.4f}\n"
        f"  - Recall   : {test_metrics['recall']:.4f}\n"
        f"  - F1-Score : {test_metrics['f1']:.4f}\n\n"
        f"TEST KARMAŞIKLIK MATRİSİ (CONFUSION MATRIX):\n"
        f"{test_metrics['confusion_matrix']}\n"
        f"Düzen: [[TN, FP], [FN, TP]]\n"
    )
    log_path.write_text(log_content, encoding="utf-8")

    print(f"\n5. Çıktılar kaydedildi:\n   - {csv_path}\n   - {log_path}")
    print("=" * 65)

    return results_df


def run_train_manual() -> pd.DataFrame:
    """Geriye dönük uyumluluk için alias fonksiyon."""
    return run()
