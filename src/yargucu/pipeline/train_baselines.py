"""Temel (baseline) modellerin eğitimi pipeline adımı."""

import sys
import time
from pathlib import Path

import pandas as pd

from yargucu.baselines.sklearn_dt import train_sklearn_dt
from yargucu.baselines.xgboost_model import train_xgboost
from yargucu.data.preprocess import run as run_preprocess


def run() -> pd.DataFrame:
    """Scikit-learn ve XGBoost modellerini eğitir, değerlendirir ve kaydeder."""
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    print("=" * 65)
    print("TEMEL MODELLERİN EĞİTİMİ (SKLEARN DT & XGBOOST BASELINES)")
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

    # 2. Sklearn Decision Tree Eğitimi
    print("\n2. Scikit-learn DecisionTreeClassifier eğitiliyor (max_depth=5, criterion='entropy', class_weight='balanced')...")
    sklearn_res = train_sklearn_dt(
        X_train, y_train, X_test, y_test, max_depth=5, criterion="entropy", class_weight="balanced", random_state=42
    )
    print(f"   - Eğitim süresi: {sklearn_res['train_time_sec']:.3f} saniye")
    print(f"   - Test Doğruluğu (Accuracy): %{sklearn_res['accuracy'] * 100:.2f} | F1: {sklearn_res['f1']:.4f}")

    # 3. XGBoost Eğitimi
    print("\n3. XGBoost XGBClassifier eğitiliyor (n_estimators=100, max_depth=5)...")
    xgb_res = train_xgboost(
        X_train, y_train, X_test, y_test, n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    print(f"   - Eğitim süresi: {xgb_res['train_time_sec']:.3f} saniye")
    print(f"   - Test Doğruluğu (Accuracy): %{xgb_res['accuracy'] * 100:.2f} | F1: {xgb_res['f1']:.4f}")

    # 4. Tablo oluşturma
    results_data = [
        {
            "model": "Sklearn Decision Tree",
            "split": "train",
            "accuracy": round(sklearn_res["train_metrics"]["accuracy"], 4),
            "precision": round(sklearn_res["train_metrics"]["precision"], 4),
            "recall": round(sklearn_res["train_metrics"]["recall"], 4),
            "f1": round(sklearn_res["train_metrics"]["f1"], 4),
            "train_time_sec": round(sklearn_res["train_time_sec"], 4),
        },
        {
            "model": "Sklearn Decision Tree",
            "split": "test",
            "accuracy": round(sklearn_res["test_metrics"]["accuracy"], 4),
            "precision": round(sklearn_res["test_metrics"]["precision"], 4),
            "recall": round(sklearn_res["test_metrics"]["recall"], 4),
            "f1": round(sklearn_res["test_metrics"]["f1"], 4),
            "train_time_sec": round(sklearn_res["train_time_sec"], 4),
        },
        {
            "model": "XGBoost",
            "split": "train",
            "accuracy": round(xgb_res["train_metrics"]["accuracy"], 4),
            "precision": round(xgb_res["train_metrics"]["precision"], 4),
            "recall": round(xgb_res["train_metrics"]["recall"], 4),
            "f1": round(xgb_res["train_metrics"]["f1"], 4),
            "train_time_sec": round(xgb_res["train_time_sec"], 4),
        },
        {
            "model": "XGBoost",
            "split": "test",
            "accuracy": round(xgb_res["test_metrics"]["accuracy"], 4),
            "precision": round(xgb_res["test_metrics"]["precision"], 4),
            "recall": round(xgb_res["test_metrics"]["recall"], 4),
            "f1": round(xgb_res["test_metrics"]["f1"], 4),
            "train_time_sec": round(xgb_res["train_time_sec"], 4),
        },
    ]
    baselines_df = pd.DataFrame(results_data)

    # 5. Çıktıları kaydetme
    tables_dir = Path("outputs/tables")
    logs_dir = Path("outputs/logs")
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "baselines_results.csv"
    baselines_df.to_csv(csv_path, index=False)

    log_path = logs_dir / "baselines.log"
    log_content = (
        f"=== TEMEL MODELLER (BASELINES) EĞİTİM LOGU ===\n"
        f"Tarih/Zaman: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"--- SKLEARN DECISION TREE ---\n"
        f"Eğitim Süresi: {sklearn_res['train_time_sec']:.4f} saniye\n"
        f"Test Accuracy : {sklearn_res['accuracy']:.4f}\n"
        f"Test Precision: {sklearn_res['precision']:.4f}\n"
        f"Test Recall   : {sklearn_res['recall']:.4f}\n"
        f"Test F1-Score : {sklearn_res['f1']:.4f}\n"
        f"Confusion Matrix:\n{sklearn_res['test_metrics']['confusion_matrix']}\n\n"
        f"--- XGBOOST CLASSIFIER ---\n"
        f"Eğitim Süresi: {xgb_res['train_time_sec']:.4f} saniye\n"
        f"Test Accuracy : {xgb_res['accuracy']:.4f}\n"
        f"Test Precision: {xgb_res['precision']:.4f}\n"
        f"Test Recall   : {xgb_res['recall']:.4f}\n"
        f"Test F1-Score : {xgb_res['f1']:.4f}\n"
        f"Confusion Matrix:\n{xgb_res['test_metrics']['confusion_matrix']}\n"
    )
    log_path.write_text(log_content, encoding="utf-8")

    print(f"\n4. Çıktılar kaydedildi:\n   - {csv_path}\n   - {log_path}")
    print("=" * 65)

    return baselines_df


def run_train_baselines() -> pd.DataFrame:
    """Geriye dönük uyumluluk için alias fonksiyon."""
    return run()
