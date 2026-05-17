from __future__ import annotations

from pathlib import Path

import pandas as pd

from retail_forecast.datasets import standardize_dataframe
from retail_forecast.exporters import build_powerbi_tables
from retail_forecast.models import train_regression_model


def _sample_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "store": ["A"] * 60,
            "item": ["SKU1"] * 60,
            "sales": [100 + i * 0.5 + (i % 7) * 2 for i in range(60)],
            "promo": [0 if i % 10 else 1 for i in range(60)],
        }
    )


def test_standardize_dataframe():
    df = standardize_dataframe(_sample_frame(), profile="rossmann")
    assert {"date", "store_id", "item_id", "target"} <= set(df.columns)
    assert df["target"].iloc[0] == 100


def test_train_regression_model():
    df = standardize_dataframe(_sample_frame(), profile="rossmann")
    bundle, metrics = train_regression_model(df)
    assert bundle.feature_columns
    assert set(metrics) == {"mae", "rmse", "mape"}


def test_build_powerbi_tables(tmp_path: Path):
    df = standardize_dataframe(_sample_frame(), profile="rossmann")
    forecast = df[["date", "store_id", "item_id"]].head(5).copy()
    forecast["forecast"] = [101, 102, 103, 104, 105]
    forecast["model_name"] = "baseline"
    paths = build_powerbi_tables(df, {"baseline": forecast}, output_dir=tmp_path)
    assert (tmp_path / "fact_history.csv").exists()
    assert (tmp_path / "fact_forecast.csv").exists()
    assert paths["inventory_recommendations"].exists()
