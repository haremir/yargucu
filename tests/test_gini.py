"""Gini indeksi (Gini Impurity) hesaplama testleri."""

import numpy as np
import pandas as pd
import pytest

from yargucu.metrics.gini import calculate_gini


def test_empty_array() -> None:
    """Boş dizi için Gini 0.0 dönmelidir."""
    assert calculate_gini([]) == 0.0
    assert calculate_gini(np.array([])) == 0.0


def test_pure_classes() -> None:
    """Tek sınıflı (tam saf) veri için Gini 0.0 dönmelidir."""
    assert calculate_gini([1, 1, 1]) == 0.0
    assert calculate_gini([0, 0, 0, 0]) == 0.0
    assert calculate_gini(["A", "A", "A"]) == 0.0


def test_binary_equal_distribution() -> None:
    """Eşit dağılımlı iki sınıf için Gini tam 0.5 dönmelidir."""
    assert calculate_gini([0, 1]) == 0.5
    assert calculate_gini([1, 0, 1, 0]) == 0.5


def test_lecture_149_samples_gini() -> None:
    """Ders notundaki 149 örnekli durum için Gini hesaplamaları.

    - Kök: 100 sınıf-1, 49 sınıf-0 -> GI = 1 - ((100/149)^2 + (49/149)^2) ≈ 0.4414
    - Sol: 50 sınıf-1, 0 sınıf-0   -> GI = 0.0
    - Sağ: 50 sınıf-1, 49 sınıf-0  -> GI = 1 - ((50/99)^2 + (49/99)^2) ≈ 0.4999
    """
    root_y = [1] * 100 + [0] * 49
    left_y = [1] * 50 + [0] * 0
    right_y = [1] * 50 + [0] * 49

    expected_root_gini = 1.0 - ((100 / 149) ** 2 + (49 / 149) ** 2)
    expected_right_gini = 1.0 - ((50 / 99) ** 2 + (49 / 99) ** 2)

    assert pytest.approx(calculate_gini(root_y), rel=1e-4) == expected_root_gini
    assert calculate_gini(left_y) == 0.0
    assert pytest.approx(calculate_gini(right_y), rel=1e-4) == expected_right_gini


def test_tennis_dataset_gini() -> None:
    """Tenis veri seti için Gini hesaplamaları."""
    tennis_all = ["Yes"] * 9 + ["No"] * 5
    tennis_weak = ["Yes"] * 6 + ["No"] * 2
    tennis_strong = ["Yes"] * 3 + ["No"] * 3

    assert pytest.approx(calculate_gini(tennis_all), rel=1e-3) == 0.459
    assert calculate_gini(tennis_weak) == 0.375
    assert calculate_gini(tennis_strong) == 0.5


def test_multiclass_gini() -> None:
    """Çok sınıflı dağılımlar için Gini hesaplaması."""
    # 3 eşit sınıf: 1 - 3 * (1/3)^2 = 2/3 ≈ 0.6667
    y_multiclass = ["A", "B", "C"] * 10
    assert pytest.approx(calculate_gini(y_multiclass), rel=1e-5) == (2.0 / 3.0)


def test_pandas_series_input() -> None:
    """Pandas Series girdisiyle doğru çalışmalıdır."""
    s = pd.Series([1, 1, 0, 0])
    assert calculate_gini(s) == 0.5
