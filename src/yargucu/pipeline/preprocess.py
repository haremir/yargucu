"""Veri ön işleme pipeline adımı."""

from yargucu.data.preprocess import run as preprocess_run


def run() -> None:
    """Veri ön işleme adımı: Verileri yükler, temizler, encode eder ve kaydeder."""
    preprocess_run()


def run_preprocess() -> None:
    """Geriye dönük uyumluluk için alias fonksiyon."""
    run()
