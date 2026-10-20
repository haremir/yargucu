"""Entropi hesaplama testleri."""

import numpy as np
import pandas as pd
import pytest

from yargucu.metrics.entropy import calculate_entropy


def test_empty_array() -> None:
    """Boş dizi için entropi 0.0 dönmelidir."""
    assert calculate_entropy([]) == 0.0
    assert calculate_entropy(np.array([])) == 0.0


def test_pure_classes() -> None:
    """Tek sınıflı (tam saf) veri için entropi 0.0 dönmelidir."""
    assert calculate_entropy([1, 1, 1, 1]) == 0.0
    assert calculate_entropy([0, 0, 0]) == 0.0
    assert calculate_entropy(["Yes", "Yes", "Yes"]) == 0.0


def test_binary_equal_distribution() -> None:
    """Eşit dağılımlı iki sınıf için entropi tam 1.0 dönmelidir."""
    assert calculate_entropy([0, 1]) == 1.0
    assert calculate_entropy([1, 0, 1, 0]) == 1.0
    assert calculate_entropy(["Yes", "No", "Yes", "No"]) == 1.0


def test_lecture_149_samples() -> None:
    """Ders notundaki 149 örnekli durum (bilgi_kazanci_ornek_cozum.pdf).

    - Kök: 100 sınıf-1, 49 sınıf-0 -> Entropi ≈ 0.913 bit
    - Sol: 50 sınıf-1, 0 sınıf-0   -> Entropi = 0.0 bit
    - Sağ: 50 sınıf-1, 49 sınıf-0  -> Entropi ≈ 1.0 bit
    """
    root_y = [1] * 100 + [0] * 49
    left_y = [1] * 50 + [0] * 0
    right_y = [1] * 50 + [0] * 49

    h_root = calculate_entropy(root_y)
    h_left = calculate_entropy(left_y)
    h_right = calculate_entropy(right_y)

    assert round(h_root, 3) == 0.913 or round(h_root, 4) == 0.9138
    assert pytest.approx(h_root, abs=0.002) == 0.913
    assert h_left == 0.0
    assert round(h_right, 2) == 1.00
    assert pytest.approx(h_right, abs=0.001) == 1.0


def test_tennis_dataset_entropy() -> None:
    """Ders notundaki 14 örnekli Tenis verisi (Karar Agaclari Algoritmasi.docx).

    - Toplam 9 Yes, 5 No -> Entropi = 0.940 bit
    - Wind=Weak (6 Yes, 2 No) -> Entropi = 0.811 bit
    - Wind=Strong (3 Yes, 3 No) -> Entropi = 1.000 bit
    """
    tennis_all = ["Yes"] * 9 + ["No"] * 5
    tennis_weak = ["Yes"] * 6 + ["No"] * 2
    tennis_strong = ["Yes"] * 3 + ["No"] * 3

    assert round(calculate_entropy(tennis_all), 2) == 0.94
    assert round(calculate_entropy(tennis_weak), 3) == 0.811
    assert calculate_entropy(tennis_strong) == 1.0


def test_multiclass_entropy() -> None:
    """Çok sınıflı dağılımlar için entropi doğrulaması."""
    # 3 eşit sınıf: Entropi = log2(3) ≈ 1.585 bit
    y_multiclass = ["A", "B", "C"] * 10
    expected_entropy = np.log2(3)
    assert pytest.approx(calculate_entropy(y_multiclass), rel=1e-5) == expected_entropy


def test_pandas_series_input() -> None:
    """Pandas Series türündeki verilerle sorunsuz çalışmalıdır."""
    s = pd.Series([1, 1, 0, 0])
    assert calculate_entropy(s) == 1.0
