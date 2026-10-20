"""Keşifsel veri analizi pipeline adımı."""

import sys
from pathlib import Path

import pandas as pd

from yargucu.data.loader import load_raw_data


def run() -> None:
    """Veri keşif adımı: Veriyi yükler, istatistikleri çıkarır ve kaydeder."""
    # Terminal çıktıları için UTF-8 desteği
    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    print("Veri seti yükleniyor...")
    df = load_raw_data()

    output_lines: list[str] = []

    def log(message: str = "") -> None:
        """Konsola basar ve rapor listesine ekler."""
        print(message)
        output_lines.append(message)

    log("=" * 60)
    log("UCI ADULT INCOME VERİ KEŞİF RAPORU (EXPLORATION REPORT)")
    log("=" * 60)

    # 1. Veri setinin boyutu
    rows, cols = df.shape
    log(f"\n1. VERİ BOYUTU:\n   - Toplam Satır Sayısı: {rows:,}")
    log(f"   - Toplam Sütun Sayısı: {cols}")

    # 2. Her sütunun veri tipi
    log("\n2. SÜTUN VERİ TİPLERİ (DTYPES):")
    for col, dtype in df.dtypes.items():
        log(f"   - {col:<16}: {dtype}")

    # 3. Eksik değer (NaN) sayıları (yalnızca > 0 olanlar)
    log("\n3. EKSİK DEĞER (NaN) ANALİZİ:")
    missing_series = df.isna().sum()
    missing_filtered = missing_series[missing_series > 0]
    if missing_filtered.empty:
        log("   - Veri setinde eksik değer bulunmamaktadır.")
    else:
        for col, count in missing_filtered.items():
            pct = (count / rows) * 100
            log(f"   - {col:<16}: {count:,} adet (%{pct:.2f})")

    # 4. Hedef değişken (income) sınıf dağılımı
    log("\n4. HEDEF DEĞİŞKEN (INCOME) SINIF DAĞILIMI:")
    if "income" in df.columns:
        counts = df["income"].value_counts(dropna=False)
        percentages = df["income"].value_counts(normalize=True, dropna=False) * 100
        dist_df = pd.DataFrame({"Count": counts, "Percentage (%)": percentages.round(2)})
        log(dist_df.to_string())
    else:
        log("   - 'income' hedef sütunu bulunamadı!")

    # 5. Kategorik sütunların benzersiz değer sayıları (nunique)
    log("\n5. KATEGORİK SÜTUNLARIN BENZERSİZ DEĞER SAYILARI (NUNIQUE):")
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        uniq_count = df[col].nunique(dropna=True)
        sample_vals = list(df[col].dropna().unique()[:4])
        sample_str = ", ".join(map(str, sample_vals))
        log(f"   - {col:<16}: {uniq_count:>2} tekil değer (Örnek: {sample_str}...)")

    # 6. Sayısal sütunların temel istatistikleri (describe)
    log("\n6. SAYISAL SÜTUNLARIN TEMEL İSTATİSTİKLERİ (DESCRIBE):")
    num_cols = df.select_dtypes(include=["number"]).columns
    if not num_cols.empty:
        desc = df.describe().T[["count", "mean", "std", "min", "50%", "max"]]
        log(desc.to_string())
    else:
        log("   - Sayısal sütun bulunamadı.")

    log("\n" + "=" * 60)

    # Raporu dosyaya yazma
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    report_file = log_dir / "explore_report.txt"
    report_file.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    log(f"Keşif raporu başarıyla kaydedildi: {report_file}")


def run_explore() -> None:
    """Geriye dönük uyumluluk için alias fonksiyon."""
    run()
