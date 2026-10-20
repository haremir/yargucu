"""Veri yükleme modülü."""

from pathlib import Path

import pandas as pd

COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def load_raw_data(data_path: str | Path | None = None) -> pd.DataFrame:
    """UCI Adult veri setinin ham verisini (train) yükler.

    Args:
        data_path: adult.data dosyasının dosya yolu. Belirtilmezse varsayılan
            konum kullanılır ("data/raw/adult.data").

    Returns:
        pd.DataFrame: Sütun adları verilmiş ve eksik değerleri NaN olarak
            işlenmiş DataFrame.
    """
    if data_path is None:
        data_path = Path("data/raw/adult.data")
    else:
        data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {data_path}")

    # Dosya virgül ve boşluk içerebilir, skipinitialspace ve na_values kullanılır
    df = pd.read_csv(
        data_path,
        header=None,
        names=COLUMN_NAMES,
        skipinitialspace=True,
        na_values=["?"],
    )

    # Hedef sütundaki olası sondaki noktaları (.) temizle (örn. <=50K. -> <=50K)
    if "income" in df.columns:
        df["income"] = df["income"].astype(str).str.rstrip(".")

    return df


def load_test_data(data_path: str | Path | None = None) -> pd.DataFrame:
    """UCI Adult test veri setini yükler.

    Args:
        data_path: adult.test dosyasının dosya yolu. Belirtilmezse varsayılan
            konum kullanılır ("data/raw/adult.test").

    Returns:
        pd.DataFrame: Sütun adları atanmış, ilk açıklama satırı atlanmış,
            noktaları temizlenmiş ve eksik değerleri NaN olarak işlenmiş test DataFrame.
    """
    if data_path is None:
        data_path = Path("data/raw/adult.test")
    else:
        data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Test veri dosyası bulunamadı: {data_path}")

    # adult.test ilk satırında açıklama metni barındırır, skiprows=1 ile atlanır
    df = pd.read_csv(
        data_path,
        header=None,
        names=COLUMN_NAMES,
        skiprows=1,
        skipinitialspace=True,
        na_values=["?"],
    )

    # Hedef değişkendeki sondaki noktaları (.) temizle (örn. '>50K.' -> '>50K')
    if "income" in df.columns:
        df["income"] = df["income"].astype(str).str.rstrip(".")

    return df
