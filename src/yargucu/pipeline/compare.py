"""Model karşılaştırma pipeline adımı."""

import sys
from pathlib import Path

import pandas as pd

from yargucu.pipeline.train_baselines import run as run_baselines
from yargucu.pipeline.train_manual import run as run_manual


def run() -> pd.DataFrame:
    """Manuel Ağaç, Sklearn DT ve XGBoost modellerinin test performanslarını karşılaştırır."""
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    manual_csv = Path("outputs/tables/manual_tree_results.csv")
    baselines_csv = Path("outputs/tables/baselines_results.csv")

    if not manual_csv.exists():
        print("Manuel ağaç sonuçları bulunamadı, eğitim çalıştırılıyor...")
        run_manual()

    if not baselines_csv.exists():
        print("Temel model (baselines) sonuçları bulunamadı, eğitim çalıştırılıyor...")
        run_baselines()

    manual_df = pd.read_csv(manual_csv)
    baselines_df = pd.read_csv(baselines_csv)

    # Yalnızca Test seti sonuçlarını kıyaslayalım
    manual_test = manual_df[manual_df["split"] == "test"]
    baselines_test = baselines_df[baselines_df["split"] == "test"]

    combined = pd.concat([manual_test, baselines_test], ignore_index=True)

    # Karşılaştırma tablosunu düzenle
    comparison_df = pd.DataFrame({
        "Model": combined["model"],
        "Accuracy (%)": (combined["accuracy"] * 100).round(2),
        "Precision (%)": (combined["precision"] * 100).round(2),
        "Recall (%)": (combined["recall"] * 100).round(2),
        "F1-Score": combined["f1"].round(4),
        "Train Time (s)": combined["train_time_sec"].round(3),
    })

    # Çıktı tablosunu CSV olarak kaydet
    tables_dir = Path("outputs/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tables_dir / "model_comparison.csv"
    comparison_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 78)
    print("                    MODEL PERFORMANS KARŞILAŞTIRMA TABLOSU                    ")
    print("=" * 78)

    # ASCII Formatlı Tablo Yazdırma
    header = f"{'Model':<25} | {'Accuracy (%)':<12} | {'Precision (%)':<13} | {'Recall (%)':<10} | {'F1-Score':<8} | {'Süre (s)':<8}"
    print(header)
    print("-" * 78)

    for _, row in comparison_df.iterrows():
        line = (
            f"{row['Model']:<25} | "
            f"{row['Accuracy (%)']:>12.2f} | "
            f"{row['Precision (%)']:>13.2f} | "
            f"{row['Recall (%)']:>10.2f} | "
            f"{row['F1-Score']:>8.4f} | "
            f"{row['Train Time (s)']:>8.3f}"
        )
        print(line)

    print("=" * 78)
    print(f"\nKarşılaştırma tablosu kaydedildi: {out_csv}")
    print("=" * 78 + "\n")

    return comparison_df


def run_compare() -> pd.DataFrame:
    """Geriye dönük uyumluluk için alias fonksiyon."""
    return run()
