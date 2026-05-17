from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .datasets import aggregate_series, infer_frequency, load_dataset
from .exporters import build_powerbi_tables
from .inventory import inventory_recommendation
from .models import (
    LinearModel,
    combine_forecasts,
    forecast_many_groups,
    seasonal_naive_forecast,
    train_regression_model,
    try_train_lstm,
    try_train_prophet,
)


@dataclass
class PipelineResult:
    raw: pd.DataFrame
    model: LinearModel | None
    metrics: dict[str, dict[str, float | int]]
    forecasts: dict[str, pd.DataFrame]
    inventory: dict[str, float] | None


def load_and_prepare(source: str | Path, profile: str | None = None) -> pd.DataFrame:
    return load_dataset(source, profile=profile)


def _evaluate_forecast(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    actual = actual.astype(float).to_numpy()
    pred = pred.astype(float).to_numpy()
    if len(actual) == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}
    mae = float(abs(actual - pred).mean())
    rmse = float(((actual - pred) ** 2).mean() ** 0.5)
    denom = pd.Series(actual).abs().clip(lower=1e-6).to_numpy()
    mape = float((abs(actual - pred) / denom).mean() * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def train_pipeline(
    source: str | Path,
    profile: str | None = None,
    model_kind: str = "ridge",
    horizon: int = 30,
    output_dir: str | Path | None = None,
    forecast_group_limit: int | None = 25,
    enable_prophet: bool = True,
    enable_lstm: bool = True,
) -> PipelineResult:
    df = load_and_prepare(source, profile=profile)
    history = df.copy().sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    metrics: dict[str, dict[str, float]] = {}
    forecasts: dict[str, pd.DataFrame] = {}

    regression_model, regression_metrics = train_regression_model(history, kind=model_kind)
    metrics["regression"] = regression_metrics

    forecasts = forecast_many_groups(
        history,
        horizon=horizon,
        max_groups=forecast_group_limit,
        enable_prophet=enable_prophet,
        enable_lstm=enable_lstm,
    )

    history_tail = history.tail(min(len(history), 30))
    inventory = inventory_recommendation(history_tail["target"])

    freq = infer_frequency(history["date"])
    for name, frame in forecasts.items():
        frame["freq"] = freq

    if output_dir is not None:
        build_powerbi_tables(history, forecasts, output_dir=Path(output_dir))

    metrics["forecast_models"] = {
        name: float(frame["forecast"].mean()) for name, frame in forecasts.items() if not frame.empty
    }
    metrics["coverage"] = {
        name: int(frame[["store_id", "item_id"]].drop_duplicates().shape[0]) if not frame.empty else 0
        for name, frame in forecasts.items()
    }

    return PipelineResult(
        raw=history,
        model=regression_model,
        metrics=metrics,
        forecasts=forecasts,
        inventory=inventory,
    )


def make_forecast(source: str | Path, profile: str | None = None, horizon: int = 30) -> pd.DataFrame:
    df = load_and_prepare(source, profile=profile)
    history = df.copy().sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    forecasts = forecast_many_groups(history, horizon=horizon, max_groups=25)
    return forecasts.get("ensemble", forecasts.get("baseline", pd.DataFrame())).copy()


def maybe_train_lstm(source: str | Path, profile: str | None = None, horizon: int = 30):
    df = load_and_prepare(source, profile=profile)
    series = aggregate_series(df)
    return try_train_lstm(series, horizon=horizon)
