"""Bilgi kazancı (Information Gain) hesaplama modülü."""

import numpy as np

from yargucu.metrics.entropy import calculate_entropy
from yargucu.metrics.gini import calculate_gini


def calculate_information_gain(
    parent_y: object,
    left_y: object,
    right_y: object,
    criterion: str = "entropy",
    parent_weight: object | None = None,
    left_weight: object | None = None,
    right_weight: object | None = None,
) -> float:
    """Sol ve sağ bölünmenin oluşturduğu Bilgi Kazancını (Information Gain) hesaplar.

    Formül:
        IG = Impurity(parent) - [ (|S_left| / |S|) * Impurity(left) + (|S_right| / |S|) * Impurity(right) ]
        Ağırlıklı durumda örnek sayıları yerine örnek ağırlıkları toplamı kullanılır.

    Args:
        parent_y: Ebeveyn düğümdeki sınıf etiketleri.
        left_y: Sol çocuk düğümdeki sınıf etiketleri.
        right_y: Sağ çocuk düğümdeki sınıf etiketleri.
        criterion: Safsızlık ölçütü ("entropy" veya "gini"). Varsayılan: "entropy".
        parent_weight: Ebeveyn örnek ağırlıkları (opsiyonel).
        left_weight: Sol çocuk örnek ağırlıkları (opsiyonel).
        right_weight: Sağ çocuk örnek ağırlıkları (opsiyonel).

    Returns:
        float: Bilgi kazancı değeri. Çocuklardan biri veya ebeveyn boşsa 0.0 döner.
    """
    parent_arr = np.asarray(parent_y)
    left_arr = np.asarray(left_y)
    right_arr = np.asarray(right_y)

    total_samples = parent_arr.size
    left_samples = left_arr.size
    right_samples = right_arr.size

    # Eğer ebeveyn boşsa veya çocuk düğümlerden biri boşsa bölünme kazanç sağlamaz
    if total_samples == 0 or left_samples == 0 or right_samples == 0:
        return 0.0

    if criterion == "gini":
        impurity_func = calculate_gini
    else:
        impurity_func = calculate_entropy

    if parent_weight is None or left_weight is None or right_weight is None:
        parent_impurity = impurity_func(parent_arr)
        left_impurity = impurity_func(left_arr)
        right_impurity = impurity_func(right_arr)

        weighted_child_impurity = (
            (left_samples / total_samples) * left_impurity
            + (right_samples / total_samples) * right_impurity
        )
    else:
        p_w = np.asarray(parent_weight, dtype=float)
        l_w = np.asarray(left_weight, dtype=float)
        r_w = np.asarray(right_weight, dtype=float)

        total_w = float(np.sum(p_w))
        left_w = float(np.sum(l_w))
        right_w = float(np.sum(r_w))

        if total_w <= 0.0 or left_w <= 0.0 or right_w <= 0.0:
            return 0.0

        parent_impurity = impurity_func(parent_arr, sample_weight=p_w)
        left_impurity = impurity_func(left_arr, sample_weight=l_w)
        right_impurity = impurity_func(right_arr, sample_weight=r_w)

        weighted_child_impurity = (
            (left_w / total_w) * left_impurity
            + (right_w / total_w) * right_impurity
        )

    gain = parent_impurity - weighted_child_impurity

    if abs(gain) < 1e-12:
        return 0.0

    return float(max(0.0, gain))
