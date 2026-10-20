"""Model performans karşılaştırma grafikleri modülü."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_model_comparison(
    csv_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Modellerin Accuracy, Precision, Recall ve F1 metriklerini karşılaştıran gruplu çubuk grafik çizer.

    Args:
        csv_path: model_comparison.csv dosya yolu. Belirtilmezse
            "outputs/tables/model_comparison.csv" kullanılır.
        output_path: Grafiğin kaydedileceği dosya yolu. Belirtilmezse
            "outputs/figures/model_comparison.png" kullanılır.

    Returns:
        Path: Kaydedilen görselin dosya yolu.
    """
    if csv_path is None:
        table_file = Path("outputs/tables/model_comparison.csv")
    else:
        table_file = Path(csv_path)

    if not table_file.exists():
        from yargucu.pipeline.compare import run as run_compare

        print("Karşılaştırma tablosu bulunamadı, önce compare pipeline'ı çalıştırılıyor...")
        run_compare()

    df = pd.read_csv(table_file)

    if output_path is None:
        save_path = Path("outputs/figures/model_comparison.png")
    else:
        save_path = Path(output_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Metrikler ve değerler (F1 skorunu yüzdeye çevirerek aynı ölçeğe alıyoruz)
    metrics = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 (%)"]
    models = df["Model"].tolist()

    # Model başına metrik değerleri matrisi (n_models, n_metrics)
    model_values = []
    for _, row in df.iterrows():
        acc = float(row["Accuracy (%)"])
        prec = float(row["Precision (%)"])
        rec = float(row["Recall (%)"])
        f1 = float(row["F1-Score"]) * 100.0  # yüzde ölçeği
        model_values.append([acc, prec, rec, f1])

    x = np.arange(len(metrics))
    total_models = len(models)
    bar_width = 0.25

    colors = ["#2b5c8f", "#e67e22", "#27ae60"]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)

    for i, model_name in enumerate(models):
        offset = (i - (total_models - 1) / 2) * bar_width
        vals = model_values[i]
        color = colors[i % len(colors)]
        bars = ax.bar(x + offset, vals, width=bar_width, label=model_name, color=color, alpha=0.9, edgecolor="black", linewidth=0.6)

        # Çubukların üstüne değerleri yazdır
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"%{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

    ax.set_title("Karar Ağacı Modelleri Performans Karşılaştırması (UCI Adult Test Seti)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight="bold")
    ax.set_ylabel("Skor (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", frameon=True, fontsize=11, shadow=True)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Grafik oluşturuldu: {save_path}")
    return save_path


def plot_comparison_chart(
    csv_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """plot_model_comparison için geriye dönük uyumluluk fonksiyonu."""
    return plot_model_comparison(csv_path, output_path)
