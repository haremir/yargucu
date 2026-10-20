"""Bilgi kazancı (Information Gain) hesaplama testleri."""

import pytest

from yargucu.metrics.entropy import calculate_entropy
from yargucu.metrics.information_gain import calculate_information_gain


def test_empty_children_returns_zero() -> None:
    """Çocuk düğümlerden biri veya ebeveyn boş olduğunda Information Gain 0.0 dönmelidir."""
    parent = [1, 0, 1, 0]
    # Sol boş
    assert calculate_information_gain(parent, [], [1, 0, 1, 0]) == 0.0
    # Sağ boş
    assert calculate_information_gain(parent, [1, 0, 1, 0], []) == 0.0
    # Ebeveyn boş
    assert calculate_information_gain([], [1], [0]) == 0.0


def test_lecture_149_samples_information_gain() -> None:
    """Ders notundaki 149 örnekli durum (bilgi_kazanci_ornek_cozum.pdf).

    - Kök: 100 sınıf-1, 49 sınıf-0 (Entropi ≈ 0.913 bit)
    - Sol: 50 sınıf-1, 0 sınıf-0 (Entropi = 0 bit)
    - Sağ: 50 sınıf-1, 49 sınıf-0 (Entropi ≈ 1 bit)
    - Bilgi Kazancı (IG) ≈ 0.249 bit
    """
    parent_y = [1] * 100 + [0] * 49
    left_y = [1] * 50 + [0] * 0
    right_y = [1] * 50 + [0] * 49

    h_parent = calculate_entropy(parent_y)
    h_left = calculate_entropy(left_y)
    h_right = calculate_entropy(right_y)

    assert round(h_parent, 3) == 0.913 or round(h_parent, 4) == 0.9138
    assert h_left == 0.0
    assert round(h_right, 2) == 1.00

    ig = calculate_information_gain(parent_y, left_y, right_y)
    assert round(ig, 3) == 0.249
    assert pytest.approx(ig, abs=0.002) == 0.249


def test_tennis_wind_information_gain() -> None:
    """Ders notundaki 14 örnekli Tenis verisi (Karar Agaclari Algoritmasi.docx).

    - Toplam 9 Yes, 5 No (Entropi = 0.940 bit)
    - Wind=Weak (6 Yes, 2 No)
    - Wind=Strong (3 Yes, 3 No)
    - Wind özelliği için IG = 0.048 bit çıkmalıdır.
    """
    tennis_parent = ["Yes"] * 9 + ["No"] * 5
    tennis_weak = ["Yes"] * 6 + ["No"] * 2
    tennis_strong = ["Yes"] * 3 + ["No"] * 3

    h_parent = calculate_entropy(tennis_parent)
    assert round(h_parent, 2) == 0.94

    ig_wind = calculate_information_gain(tennis_parent, tennis_weak, tennis_strong)
    assert round(ig_wind, 3) == 0.048
    assert pytest.approx(ig_wind, abs=0.001) == 0.048


def test_perfect_split_information_gain() -> None:
    """Tam ayrım durumunda bilgi kazancı ebeveynin tüm belirsizliğini yok etmelidir."""
    parent = [1, 1, 1, 1, 0, 0, 0, 0]  # Entropi = 1.0
    left = [1, 1, 1, 1]  # Entropi = 0.0
    right = [0, 0, 0, 0]  # Entropi = 0.0

    ig = calculate_information_gain(parent, left, right)
    assert ig == 1.0


def test_gini_criterion_information_gain() -> None:
    """Gini safsızlık ölçütü ile bilgi kazancı hesaplanabilmelidir."""
    parent = [1, 1, 0, 0]  # Gini = 0.5
    left = [1, 1]  # Gini = 0.0
    right = [0, 0]  # Gini = 0.0

    ig_gini = calculate_information_gain(parent, left, right, criterion="gini")
    assert ig_gini == 0.5
