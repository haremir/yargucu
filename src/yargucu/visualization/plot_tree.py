"""Karar ağacı yapısını görselleştirme modülü."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from yargucu.tree.decision_tree import DecisionTreeClassifier
from yargucu.tree.node import Node


def _build_dot_string(node: Node | None, feature_names: list[str] | None, node_id: list[int]) -> str:
    """Ağacı Graphviz .dot formatına dönüştürür."""
    if node is None:
        return ""

    curr_id = node_id[0]
    node_id[0] += 1

    if node.is_leaf:
        label = f"Sınıf: {node.value}"
        color = "#d4edda" if str(node.value) == "0" else "#ffeeba"
        lines = [f'    {curr_id} [label="{label}", shape=box, style="filled,rounded", fillcolor="{color}"];']
    else:
        feat = (
            feature_names[node.feature_index]
            if feature_names and node.feature_index is not None and node.feature_index < len(feature_names)
            else f"X[{node.feature_index}]"
        )
        label = f"{feat}\\n<= {node.threshold:.2f}"
        lines = [f'    {curr_id} [label="{label}", shape=box, style="filled,rounded", fillcolor="#cce5ff"];']

        if node.left is not None:
            left_id = node_id[0]
            lines.append(f'    {curr_id} -> {left_id} [label="Evet", color="green"];')
            lines.append(_build_dot_string(node.left, feature_names, node_id))

        if node.right is not None:
            right_id = node_id[0]
            lines.append(f'    {curr_id} -> {right_id} [label="Hayır", color="red"];')
            lines.append(_build_dot_string(node.right, feature_names, node_id))

    return "\n".join(lines)


def plot_manual_tree(
    model: Any = None,
    feature_names: list[str] | None = None,
    max_depth: int = 3,
    output_path: str | Path | None = None,
) -> Path:
    """Manuel eğitilen Karar Ağacını görselleştirir ve PNG olarak kaydeder.

    Args:
        model: Eğitilmiş DecisionTreeClassifier modeli. Belirtilmezse max_depth=3
            ile eğitilip çizilir.
        feature_names: Öznitelik isimleri listesi.
        max_depth: Görselleştirilecek maksimum derinlik seviyesi.
        output_path: Görselin kaydedileceği dosya yolu. Belirtilmezse
            "outputs/figures/manual_tree.png" kullanılır.

    Returns:
        Path: Kaydedilen görselin dosya yolu.
    """
    if output_path is None:
        save_path = Path("outputs/figures/manual_tree.png")
    else:
        save_path = Path(output_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Model veya özellik isimleri verilmemişse hazır veriden eğit
    if model is None or feature_names is None:
        train_path = Path("data/processed/train.csv")
        if not train_path.exists():
            from yargucu.data.preprocess import run as run_preprocess

            run_preprocess()

        train_df = pd.read_csv(train_path)
        feature_names = [c for c in train_df.columns if c != "income"]
        X_train = train_df.drop(columns=["income"]).values
        y_train = train_df["income"].values

        model = DecisionTreeClassifier(max_depth=max_depth, criterion="entropy")
        model.fit(X_train, y_train)

    # 1. Graphviz .dot dosyasını da ek olarak kaydet
    dot_content = "digraph DecisionTree {\n" + _build_dot_string(model.root_, feature_names, [0]) + "\n}\n"
    dot_path = save_path.with_suffix(".dot")
    dot_path.write_text(dot_content, encoding="utf-8")

    # 2. Matplotlib ile taşınabilir ve yüksek çözünürlüklü çizim
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.set_axis_off()
    ax.set_title(
        f"Manuel Karar Ağacı Şeması (İlk {max_depth} Derinlik Seviyesi)",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    def get_node_label(node: Node) -> str:
        if node.is_leaf:
            cls_name = "<=50K (0)" if str(node.value) == "0" else ">50K (1)"
            return f"Yaprak Düğüm\nTahmin: {cls_name}"
        feat_name = (
            feature_names[node.feature_index]
            if feature_names and node.feature_index is not None and node.feature_index < len(feature_names)
            else f"X[{node.feature_index}]"
        )
        if len(feat_name) > 24:
            feat_name = feat_name[:22] + ".."
        return f"{feat_name}\n<= {node.threshold:.2f}"

    def draw_node(
        node: Node | None,
        x: float,
        y: float,
        dx: float,
        dy: float,
        curr_depth: int,
    ) -> None:
        if node is None or curr_depth > max_depth:
            return

        text = get_node_label(node)
        color = "#d4edda" if node.is_leaf else "#cce5ff"
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold" if node.is_leaf else "normal",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": color, "edgecolor": "#2c3e50", "lw": 1.2},
        )

        if not node.is_leaf and curr_depth < max_depth:
            # Sol çocuk (Evet / True)
            if node.left is not None:
                xl, yl = x - dx, y - dy
                ax.plot([x, xl], [y - 0.045, yl + 0.045], color="#27ae60", lw=1.5)
                ax.text(
                    (x + xl) / 2 - 0.02,
                    (y + yl) / 2,
                    "Evet",
                    fontsize=8,
                    fontweight="bold",
                    color="#27ae60",
                )
                draw_node(node.left, xl, yl, dx / 2, dy, curr_depth + 1)

            # Sağ çocuk (Hayır / False)
            if node.right is not None:
                xr, yr = x + dx, y - dy
                ax.plot([x, xr], [y - 0.045, yr + 0.045], color="#c0392b", lw=1.5)
                ax.text(
                    (x + xr) / 2 + 0.02,
                    (y + yr) / 2,
                    "Hayır",
                    fontsize=8,
                    fontweight="bold",
                    color="#c0392b",
                )
                draw_node(node.right, xr, yr, dx / 2, dy, curr_depth + 1)

    draw_node(model.root_, x=0.5, y=0.92, dx=0.25, dy=0.25, curr_depth=0)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Grafik oluşturuldu: {save_path}")
    print(f"DOT dosyası kaydedildi: {dot_path}")
    return save_path


def plot_tree_structure(
    model: Any = None,
    feature_names: list[str] | None = None,
    max_depth: int = 3,
    output_path: str | Path | None = None,
) -> Path:
    """plot_manual_tree için geriye dönük uyumluluk fonksiyonu."""
    return plot_manual_tree(model, feature_names, max_depth, output_path)
