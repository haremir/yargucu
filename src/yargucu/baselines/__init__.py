"""Temel (baseline) modeller paketi."""

from yargucu.baselines.sklearn_dt import get_sklearn_model, train_sklearn_dt
from yargucu.baselines.xgboost_model import get_xgboost_model, train_xgboost

__all__ = [
    "get_sklearn_model",
    "get_xgboost_model",
    "train_sklearn_dt",
    "train_xgboost",
]
