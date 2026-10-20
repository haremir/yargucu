"""Görselleştirme pipeline adımı."""

import sys
from pathlib import Path

from yargucu.visualization.plot_comparison import plot_model_comparison
from yargucu.visualization.plot_entropy import plot_entropy_curve
from yargucu.visualization.plot_tree import plot_manual_tree


def run() -> list[Path]:
    """Tüm grafik ve görselleştirme fonksiyonlarını çalıştırır."""
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    print("=" * 65)
    print("GÖRSELLEŞTİRME VE GRAFİK ÜRETİMİ (VISUALIZATION PIPELINE)")
    print("=" * 65)

    generated_files: list[Path] = []

    print("\n1. Entropi ve Gini karşılaştırma eğrisi çiziliyor...")
    f1 = plot_entropy_curve()
    generated_files.append(f1)

    print("\n2. Model performans karşılaştırma grafiği çiziliyor...")
    f2 = plot_model_comparison()
    generated_files.append(f2)

    print("\n3. Manuel Karar Ağacı yapısı çiziliyor (ilk 3 seviye)...")
    f3 = plot_manual_tree(max_depth=3)
    generated_files.append(f3)

    print("\n" + "=" * 65)
    print("Görselleştirme tamamlandı! Üretilen dosyalar:")
    for f in generated_files:
        print(f"  - {f}")
    print("=" * 65)

    return generated_files


def run_visualize() -> list[Path]:
    """Geriye dönük uyumluluk için alias fonksiyon."""
    return run()
