"""Entropi ve Gini safsızlık (impurity) eğrilerini görselleştirme modülü."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_entropy_curve(output_path: str | Path | None = None) -> Path:
    """İkili sınıflandırma için Entropi ve Gini kirlilik eğrilerini çizer ve kaydeder.

    Args:
        output_path: Grafiğin kaydedileceği dosya yolu. Belirtilmezse
            "outputs/figures/entropy_vs_gini.png" kullanılır.

    Returns:
        Path: Kaydedilen görsel dosyasının yolu.
    """
    if output_path is None:
        save_path = Path("outputs/figures/entropy_vs_gini.png")
    else:
        save_path = Path(output_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # p olasılığı [0, 1] aralığı
    p = np.linspace(0.0001, 0.9999, 1000)

    # Entropi: -p*log2(p) - (1-p)*log2(1-p)
    entropy = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))

    # Gini: 1 - (p^2 + (1-p)^2) = 2*p*(1-p)
    gini = 1.0 - (p**2 + (1.0 - p) ** 2)

    # Karşılaştırma için ölçeklendirilmiş Gini (2 * Gini)
    scaled_gini = 2.0 * gini

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    ax.plot(p, entropy, label="Shannon Entropisi $H(p)$ (Maks: 1.0)", color="#1f77b4", linewidth=2.5)
    ax.plot(p, gini, label="Gini Safsızlığı $GI(p)$ (Maks: 0.5)", color="#ff7f0e", linewidth=2.5)
    ax.plot(
        p,
        scaled_gini,
        label="Ölçeklenmiş Gini $2 \\times GI(p)$ (Maks: 1.0)",
        color="#2ca02c",
        linestyle="--",
        linewidth=1.8,
        alpha=0.85,
    )

    # Zirve noktaları ve dikey çizgi
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.7)
    ax.scatter([0.5, 0.5], [1.0, 0.5], color=["#1f77b4", "#ff7f0e"], zorder=5, s=60)
    ax.annotate(
        "Maksimum Belirsizlik\n(p = 0.5)",
        xy=(0.5, 0.5),
        xytext=(0.55, 0.35),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
        fontsize=10,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "yellow", "alpha": 0.3},
    )

    ax.set_title("İkili Sınıflandırma: Entropi ve Gini Safsızlık Eğrileri", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Pozitif Sınıf Olasılığı ($p$)", fontsize=12)
    ax.set_ylabel("Safsızlık / Kirlilik Değeri", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center", frameon=True, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Grafik oluşturuldu: {save_path}")
    return save_path


def plot_entropy_vs_gini(output_path: str | Path | None = None) -> Path:
    """plot_entropy_curve için geriye dönük uyumluluk fonksiyonu."""
    return plot_entropy_curve(output_path)
