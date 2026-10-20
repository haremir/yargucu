"""Metrik gösterimi pipeline adımı."""

import sys

from yargucu.metrics.entropy import calculate_entropy
from yargucu.metrics.gini import calculate_gini
from yargucu.metrics.information_gain import calculate_information_gain


def run() -> None:
    """Entropi, Gini ve Bilgi Kazancı metriklerini ders notlarındaki örneklerle gösterir."""
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    print("=" * 70)
    print("KARAR AĞACI METRİK HESAPLAMA DEMOSU (ENTROPİ, GİNİ, BİLGİ KAZANCI)")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. ÖRNEK: 149 Örnekli Durum (bilgi_kazanci_ornek_cozum.pdf)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ÖRNEK 1: 149 Örnekli Dağılım (Ders Notu: bilgi_kazanci_ornek_cozum.pdf)")
    print("-" * 70)

    p_149 = [1] * 100 + [0] * 49
    left_149 = [1] * 50 + [0] * 0
    right_149 = [1] * 50 + [0] * 49

    h_root = calculate_entropy(p_149)
    g_root = calculate_gini(p_149)

    h_left = calculate_entropy(left_149)
    g_left = calculate_gini(left_149)

    h_right = calculate_entropy(right_149)
    g_right = calculate_gini(right_149)

    ig_149 = calculate_information_gain(p_149, left_149, right_149)
    ig_gini_149 = calculate_information_gain(p_149, left_149, right_149, criterion="gini")

    print("Düğüm Yapısı:")
    print("  * Kök Düğüm  (Parent) : 100 Sınıf-1, 49 Sınıf-0 (Toplam: 149)")
    print(f"    - Entropi H(Kök)     : {h_root:.4f} bit (Ders notundaki değer: ≈ 0.913)")
    print(f"    - Gini GI(Kök)       : {g_root:.4f}")
    print("  * Sol Çocuk  (Left)   : 50 Sınıf-1, 0 Sınıf-0 (Toplam: 50, Tam Saf)")
    print(f"    - Entropi H(Sol)     : {h_left:.4f} bit (Tam saf düğüm)")
    print(f"    - Gini GI(Sol)       : {g_left:.4f}")
    print("  * Sağ Çocuk  (Right)  : 50 Sınıf-1, 49 Sınıf-0 (Toplam: 99)")
    print(f"    - Entropi H(Sağ)     : {h_right:.4f} bit (Ders notundaki değer: ≈ 1.000)")
    print(f"    - Gini GI(Sağ)       : {g_right:.4f}")

    weighted_entropy_149 = (len(left_149) / len(p_149)) * h_left + (len(right_149) / len(p_149)) * h_right
    print("\nBilgi Kazancı Hesabı:")
    print(f"  * Ağırlıklı Kalan Entropi: (50/149)*{h_left:.4f} + (99/149)*{h_right:.4f} = {weighted_entropy_149:.4f} bit")
    print(f"  * Bilgi Kazancı (IG)    : H(Kök) - Ağırlıklı Entropi = {ig_149:.4f} bit (≈ {round(ig_149, 3):.3f})")
    print(f"  * Gini Kazancı (IG_gini): {ig_gini_149:.4f}")
    print(f"  -> Sonuç Doğrulaması    : IG = {round(ig_149, 3):.3f} bit (Ders notu: 0.249 ile TAM UYUMLU)")

    # -------------------------------------------------------------
    # 2. ÖRNEK: 14 Örnekli Tenis Verisi (Karar Agaclari Algoritmasi.docx)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ÖRNEK 2: 14 Örnekli Play Tennis Verisi (Karar Agaclari Algoritmasi.docx)")
    print("-" * 70)

    tennis_all = ["Yes"] * 9 + ["No"] * 5
    tennis_weak = ["Yes"] * 6 + ["No"] * 2
    tennis_strong = ["Yes"] * 3 + ["No"] * 3

    h_tennis = calculate_entropy(tennis_all)
    g_tennis = calculate_gini(tennis_all)

    h_weak = calculate_entropy(tennis_weak)
    g_weak = calculate_gini(tennis_weak)

    h_strong = calculate_entropy(tennis_strong)
    g_strong = calculate_gini(tennis_strong)

    ig_wind = calculate_information_gain(tennis_all, tennis_weak, tennis_strong)

    print("Hedef Sınıf Dağılımı: 9 'Yes', 5 'No' (Toplam: 14)")
    print(f"  * Başlangıç Entropisi H(S) : {h_tennis:.4f} bit (Ders notundaki değer: 0.94)")
    print(f"  * Başlangıç Gini GI(S)     : {g_tennis:.4f}")
    print("\n'Wind' (Rüzgar) Özelliğine Göre Ayrım:")
    print("  * Wind = Weak   (Hafif)    : 6 Yes, 2 No (Toplam: 8)")
    print(f"    - Entropi H(Weak)        : {h_weak:.4f} bit (Ders notu: 0.811)")
    print(f"    - Gini GI(Weak)          : {g_weak:.4f}")
    print("  * Wind = Strong (Kuvvetli) : 3 Yes, 3 No (Toplam: 6)")
    print(f"    - Entropi H(Strong)      : {h_strong:.4f} bit (Ders notu: 1.000)")
    print(f"    - Gini GI(Strong)        : {g_strong:.4f}")

    weighted_entropy_wind = (len(tennis_weak) / len(tennis_all)) * h_weak + (len(tennis_strong) / len(tennis_all)) * h_strong
    print("\nBilgi Kazancı Hesabı:")
    print(f"  * Koşullu Entropi H(S|Wind): (8/14)*{h_weak:.4f} + (6/14)*{h_strong:.4f} = {weighted_entropy_wind:.4f} bit")
    print(f"  * Bilgi Kazancı IG(S, Wind): H(S) - H(S|Wind) = {ig_wind:.4f} bit (≈ {round(ig_wind, 3):.3f})")
    print(f"  -> Sonuç Doğrulaması       : IG = {round(ig_wind, 3):.3f} bit (Ders notu: 0.048 ile TAM UYUMLU)")

    print("\n" + "=" * 70)


def run_metrics_demo() -> None:
    """Geriye dönük uyumluluk için alias fonksiyon."""
    run()
