# 1 → explore
# 2 → preprocess
# 3 → metrics-demo
# 4 → train-manual
# 5 → train-sklearn
# 6 → train-xgboost
# 7 → compare
# 8 → visualize
# 9 → optimize

import typer

app = typer.Typer(
    name="yargucu",
    help="Yargucu: Karar Ağacı Algoritması ve Kıyaslama CLI Aracı",
)


@app.command()
def explore() -> None:
    """Veri setini keşfet ve özet istatistikleri göster."""
    from yargucu.pipeline.explore import run

    run()


@app.command()
def preprocess() -> None:
    """Ham veriyi temizle, eksik değerleri işle ve dönüştür."""
    from yargucu.pipeline.preprocess import run

    run()


@app.command(name="metrics-demo")
def metrics_demo() -> None:
    """Entropi, Gini ve Bilgi Kazancı metriklerini örnek verilerle göster."""
    from yargucu.pipeline.metrics_demo import run

    run()


@app.command(name="train-manual")
def train_manual() -> None:
    """Sıfırdan yazılan Karar Ağacı modelini eğit."""
    from yargucu.pipeline.train_manual import run

    run()


@app.command(name="train-baselines")
def train_baselines() -> None:
    """Scikit-learn ve XGBoost temel modellerini eğit."""
    from yargucu.pipeline.train_baselines import run

    run()


@app.command(name="train-sklearn")
def train_sklearn() -> None:
    """Scikit-learn Karar Ağacı modelini eğit."""
    from yargucu.pipeline.train_baselines import run

    run()


@app.command(name="train-xgboost")
def train_xgboost() -> None:
    """XGBoost modelini eğit."""
    from yargucu.pipeline.train_baselines import run

    run()


@app.command()
def compare() -> None:
    """Tüm modellerin performanslarını ve metriklerini karşılaştır."""
    from yargucu.pipeline.compare import run

    run()


@app.command()
def visualize() -> None:
    """Karar ağacını ve karşılaştırma grafiklerini görselleştir."""
    from yargucu.pipeline.visualize import run

    run()


@app.command()
def optimize() -> None:
    """Ağaç derinliği ve safsızlık ölçütleri hiperparametre optimizasyon deneylerini çalıştır."""
    from yargucu.pipeline.optimize import run

    run()


if __name__ == "__main__":
    app()
