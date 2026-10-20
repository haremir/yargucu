"""Hiperparametre optimizasyonu ve budama (pruning/depth) deneyleri pipeline adımı."""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from yargucu.data.preprocess import run as run_preprocess
from yargucu.evaluation.metrics import evaluate_predictions
from yargucu.tree.decision_tree import DecisionTreeClassifier


def run() -> pd.DataFrame:
    """Hiperparametre optimizasyonu deneylerini (Derinlik ve Kirlilik Kriteri) yürütür."""
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    print("=" * 72)
    print("HİPERPARAMETRE OPTİMİZASYONU VE AĞAÇ DERİNLİĞİ ANALİZİ (OPTIMIZE PIPELINE)")
    print("=" * 72)

    train_path = Path("data/processed/train.csv")
    test_path = Path("data/processed/test.csv")

    if not train_path.exists() or not test_path.exists():
        print("İşlenmiş veri dosyaları bulunamadı, ön işleme adımı çalıştırılıyor...")
        run_preprocess()

    print("\n1. Veri setleri yükleniyor...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["income"]).values
    y_train = train_df["income"].values
    X_test = test_df.drop(columns=["income"]).values
    y_test = test_df["income"].values

    # -------------------------------------------------------------
    # DENEY 1: Ağaç Derinliği (max_depth) vs Performans (Overfitting)
    # -------------------------------------------------------------
    depth_list = [3, 5, 7, 10, 15]
    print("\n2. DENEY 1: Ağaç Derinliğinin Performans ve Aşırı Öğrenmeye (Overfitting) Etkisi")
    print("-" * 72)
    print(f"{'max_depth':<10} | {'Eğitim Doğruluğu (%)':<22} | {'Test Doğruluğu (%)':<20} | {'F1-Score':<10} | {'Süre (s)':<8}")
    print("-" * 72)

    results_data: list[dict] = []
    train_acc_list: list[float] = []
    test_acc_list: list[float] = []

    for depth in depth_list:
        start_t = time.perf_counter()
        model = DecisionTreeClassifier(max_depth=depth, criterion="entropy")
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - start_t

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_metrics = evaluate_predictions(y_train, train_preds)
        test_metrics = evaluate_predictions(y_test, test_preds)

        tr_acc = train_metrics["accuracy"] * 100.0
        te_acc = test_metrics["accuracy"] * 100.0
        te_f1 = test_metrics["f1"]

        train_acc_list.append(tr_acc)
        test_acc_list.append(te_acc)

        print(f"{depth:<10} | {tr_acc:>20.2f}% | {te_acc:>18.2f}% | {te_f1:>10.4f} | {fit_time:>8.2f}")

        results_data.append({
            "experiment": "depth_tuning",
            "max_depth": depth,
            "criterion": "entropy",
            "train_accuracy": round(train_metrics["accuracy"], 4),
            "test_accuracy": round(test_metrics["accuracy"], 4),
            "test_precision": round(test_metrics["precision"], 4),
            "test_recall": round(test_metrics["recall"], 4),
            "test_f1": round(test_metrics["f1"], 4),
            "train_time_sec": round(fit_time, 4),
        })

    # -------------------------------------------------------------
    # DENEY 2: Kirlilik Ölçütlerinin Karşılaştırılması (Entropy vs Gini)
    # -------------------------------------------------------------
    print("\n3. DENEY 2: Safsızlık Ölçütleri Karşılaştırması (max_depth=5 Sabit)")
    print("-" * 72)
    print(f"{'Ölçüt':<10} | {'Eğitim Doğruluğu (%)':<22} | {'Test Doğruluğu (%)':<20} | {'F1-Score':<10} | {'Süre (s)':<8}")
    print("-" * 72)

    for criterion in ["entropy", "gini"]:
        start_t = time.perf_counter()
        model = DecisionTreeClassifier(max_depth=5, criterion=criterion)
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - start_t

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_metrics = evaluate_predictions(y_train, train_preds)
        test_metrics = evaluate_predictions(y_test, test_preds)

        tr_acc = train_metrics["accuracy"] * 100.0
        te_acc = test_metrics["accuracy"] * 100.0
        te_f1 = test_metrics["f1"]

        print(f"{criterion:<10} | {tr_acc:>20.2f}% | {te_acc:>18.2f}% | {te_f1:>10.4f} | {fit_time:>8.2f}")

        results_data.append({
            "experiment": "criterion_comparison",
            "max_depth": 5,
            "criterion": criterion,
            "train_accuracy": round(train_metrics["accuracy"], 4),
            "test_accuracy": round(test_metrics["accuracy"], 4),
            "test_precision": round(test_metrics["precision"], 4),
            "test_recall": round(test_metrics["recall"], 4),
            "test_f1": round(test_metrics["f1"], 4),
            "train_time_sec": round(fit_time, 4),
        })

    # Sonuçları CSV olarak kaydet
    tables_dir = Path("outputs/tables")
    figures_dir = Path("outputs/figures")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results_data)
    results_csv = tables_dir / "hyperparameter_results.csv"
    results_df.to_csv(results_csv, index=False)

    # Çizgi Grafik: Derinlik vs Performans (Overfitting Analizi)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    ax.plot(
        depth_list,
        train_acc_list,
        marker="o",
        linewidth=2.5,
        color="#2980b9",
        label="Eğitim Doğruluğu (Train Accuracy)",
    )
    ax.plot(
        depth_list,
        test_acc_list,
        marker="s",
        linewidth=2.5,
        color="#e74c3c",
        label="Test Doğruluğu (Test Accuracy)",
    )

    # En iyi test doğruluğunu işaretle
    best_idx = int(pd.Series(test_acc_list).idxmax())
    best_depth = depth_list[best_idx]
    best_score = test_acc_list[best_idx]

    ax.scatter([best_depth], [best_score], color="#27ae60", s=130, zorder=5, edgecolor="black")
    ax.annotate(
        f"Optimal Derinlik: {best_depth}\n(Test Acc: %{best_score:.2f})",
        xy=(best_depth, best_score),
        xytext=(best_depth - 1.5, best_score - 1.2),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
        fontsize=10,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#d4edda", "edgecolor": "#27ae60", "alpha": 0.9},
    )

    # Overfitting bölgesini işaretle
    ax.annotate(
        "Aşırı Öğrenme (Overfitting):\nEğitim skoru artarken\nTest skoru düşüyor",
        xy=(15, test_acc_list[-1]),
        xytext=(11.5, test_acc_list[-1] - 1.6),
        arrowprops={"arrowstyle": "->", "color": "#c0392b", "lw": 1.2},
        fontsize=9.5,
        color="#c0392b",
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f8d7da", "edgecolor": "#c0392b", "alpha": 0.9},
    )

    ax.set_title("Karar Ağacı Derinlik Optimizasyonu ve Aşırı Öğrenme (Overfitting) Analizi", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Maksimum Ağaç Derinliği (max_depth)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Model Doğruluğu (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(depth_list)
    ax.legend(loc="lower right", frameon=True, fontsize=10.5)
    ax.grid(True, linestyle="--", alpha=0.7)

    fig.tight_layout()
    analysis_fig = figures_dir / "hyperparameter_analysis.png"
    fig.savefig(analysis_fig, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 72)
    print("Deneyler tamamlandı!")
    print(f"  - Sonuç tablosu: {results_csv}")
    print(f"  - Analiz grafiği: {analysis_fig}")
    print("=" * 72)

    return results_df
