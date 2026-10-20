"""Veri ön işleme ve dönüştürme fonksiyonları."""

import sys
from pathlib import Path

import pandas as pd

from yargucu.data.loader import load_raw_data, load_test_data

_TRAIN_COLUMNS: list[str] | None = None
_TRAIN_MODES: dict[str, object] | None = None


def clean_and_encode(
    df: pd.DataFrame,
    is_train: bool = True,
    train_columns: list[str] | None = None,
    train_modes: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Veriyi temizler, eksik değerleri doldurur ve kategorik değişkenleri One-Hot encode eder.

    Args:
        df: Temizlenecek ve dönüştürülecek ham DataFrame.
        is_train: Verinin eğitim seti olup olmadığını belirtir. True ise mod ve
            sütun şablonu eğitim setinden öğrenilir ve saklanır.
        train_columns: Test seti için kullanılacak eğitim seti sütun listesi (opsiyonel).
        train_modes: Test setinde eksik değerleri doldurmak için eğitim modları (opsiyonel).

    Returns:
        pd.DataFrame: Ön işleme tabi tutulmuş ve sütunları hizalanmış DataFrame.
    """
    global _TRAIN_COLUMNS, _TRAIN_MODES

    data = df.copy()

    # 1. 'fnlwgt' sütununu çıkar
    if "fnlwgt" in data.columns:
        data = data.drop(columns=["fnlwgt"])

    # 2. Eksik (NaN) değerleri mod ile doldur
    if is_train:
        computed_modes = {
            col: data[col].mode(dropna=True)[0]
            for col in data.columns
            if not data[col].mode(dropna=True).empty
        }
        _TRAIN_MODES = computed_modes
        modes = computed_modes
    else:
        if train_modes is not None:
            modes = train_modes
        elif _TRAIN_MODES is not None:
            modes = _TRAIN_MODES
        else:
            modes = {
                col: data[col].mode(dropna=True)[0]
                for col in data.columns
                if not data[col].mode(dropna=True).empty
            }

    data = data.fillna(modes)

    # 3. 'income' hedef sütununu 0 ve 1 olarak encode et (<=50K: 0, >50K: 1)
    if "income" in data.columns and not pd.api.types.is_numeric_dtype(data["income"]):
        cleaned_income = data["income"].astype(str).str.rstrip(".")
        income_map = {"<=50K": 0, ">50K": 1}
        data["income"] = cleaned_income.map(income_map).astype(int)

    # 4. Kategorik sütunları One-Hot Encoding ile encode et (drop_first=True)
    data = pd.get_dummies(data, drop_first=True, dtype=int)

    # 5. Eğitim ve test sütunlarının birebir aynı olmasını sağla (reindex)
    if is_train:
        _TRAIN_COLUMNS = list(data.columns)
    else:
        target_cols = train_columns if train_columns is not None else _TRAIN_COLUMNS
        if target_cols is not None:
            data = data.reindex(columns=target_cols, fill_value=0)

    return data


def clean_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """clean_and_encode için geriye dönük uyumluluk fonksiyonu."""
    return clean_and_encode(df, is_train=is_train)


def run(
    train_path: str | Path | None = None,
    test_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Eğitim ve test verilerini yükler, ön işler ve CSV olarak kaydeder."""
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    print("=" * 65)
    print("VERİ ÖN İŞLEME VE DÖNÜŞTÜRME (PREPROCESSING)")
    print("=" * 65)

    print("\n1. Ham veriler yükleniyor...")
    train_raw = load_raw_data(train_path)
    test_raw = load_test_data(test_path)

    raw_train_shape = train_raw.shape
    raw_test_shape = test_raw.shape
    print(f"   - Ham Eğitim Seti Boyutu: {raw_train_shape[0]:,} satır, {raw_train_shape[1]} sütun")
    print(f"   - Ham Test Seti Boyutu  : {raw_test_shape[0]:,} satır, {raw_test_shape[1]} sütun")

    print("\n2. Ön işleme adımları uygulanıyor:")
    print("   - 'fnlwgt' sütunu kaldırılıyor.")
    print("   - Eksik değerler (NaN) eğitim seti mod değerleri ile dolduruluyor.")
    print("   - Hedef değişken ('income') encode ediliyor (<=50K -> 0, >50K -> 1).")
    print("   - Kategorik değişkenler One-Hot Encoding (drop_first=True) ile dönüştürülüyor.")

    # Eğitim ve test setlerini dönüştür
    train_processed = clean_and_encode(train_raw, is_train=True)
    test_processed = clean_and_encode(
        test_raw,
        is_train=False,
        train_columns=list(train_processed.columns),
    )

    # İstatistikler
    original_feature_count = raw_train_shape[1] - 2  # fnlwgt ve income hariç
    total_new_cols = train_processed.shape[1]
    one_hot_cols_count = total_new_cols - 1 - 5  # income ve 5 sayısal hariç

    print("\n3. Sütun Dönüşüm Özeti:")
    print(f"   - Orijinal kategorik özellik sayısı  : {original_feature_count - 5}")
    print(f"   - One-Hot ile türetilen kukla sütunlar: {one_hot_cols_count} adet")
    print(f"   - Toplam yeni sütun sayısı (Özellikler + Hedef): {total_new_cols} sütun")

    print("\n4. İşlem Sonrası Veri Boyutları:")
    print(f"   - İşlenmiş Eğitim Seti: {train_processed.shape[0]:,} satır, {train_processed.shape[1]} sütun")
    print(f"   - İşlenmiş Test Seti  : {test_processed.shape[0]:,} satır, {test_processed.shape[1]} sütun")

    # Sütun uyumu doğrulaması
    columns_match = list(train_processed.columns) == list(test_processed.columns)
    print(f"   - Eğitim ve Test sütun yapısı birebir aynı mı: {'EVET' if columns_match else 'HAYIR'}")

    # Dosyalara kaydetme
    if output_dir is None:
        out_path = Path("data/processed")
    else:
        out_path = Path(output_dir)

    out_path.mkdir(parents=True, exist_ok=True)
    train_out_file = out_path / "train.csv"
    test_out_file = out_path / "test.csv"

    train_processed.to_csv(train_out_file, index=False)
    test_processed.to_csv(test_out_file, index=False)

    print(f"\n5. Dosyalar kaydedildi:\n   - {train_out_file}\n   - {test_out_file}")
    print("=" * 65)

    return train_processed, test_processed
