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
    seasonal_naive_forecast,
    train_regression_model,
    try_train_lstm,
    try_train_prophet,
)


@dataclass
class PipelineResult:
    raw: pd.DataFrame
    metrics: dict[str, dict[str, float]]
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
    series_level: bool = True,
    output_dir: str | Path | None = None,
) -> PipelineResult:
    df = load_and_prepare(source, profile=profile)
    series = aggregate_series(df) if series_level else df.copy()

    metrics: dict[str, dict[str, float]] = {}
    forecasts: dict[str, pd.DataFrame] = {}

    regression_model, regression_metrics = train_regression_model(series, kind=model_kind)
    metrics["regression"] = regression_metrics

    baseline_forecast = seasonal_naive_forecast(series, horizon=horizon)
    forecasts["baseline"] = baseline_forecast.rename(columns={"ds": "date", "yhat": "forecast"})

    prophet_forecast = None
    try:
        _, prophet_forecast = try_train_prophet(series, horizon=horizon)
        forecasts["prophet"] = prophet_forecast.rename(columns={"ds": "date", "yhat": "forecast"})
    except Exception:
        prophet_forecast = None

    lstm_forecast = None
    try:
        _, lstm_forecast = try_train_lstm(series, horizon=horizon)
        forecasts["lstm"] = lstm_forecast.rename(columns={"ds": "date", "yhat": "forecast"})
    except Exception:
        lstm_forecast = None

    if prophet_forecast is not None and lstm_forecast is not None:
        ensemble = combine_forecasts(prophet_forecast, lstm_forecast)
        forecasts["ensemble"] = ensemble.rename(columns={"ds": "date", "yhat": "forecast"})
    elif prophet_forecast is not None:
        forecasts["ensemble"] = prophet_forecast.rename(columns={"ds": "date", "yhat": "forecast"})
    elif lstm_forecast is not None:
        forecasts["ensemble"] = lstm_forecast.rename(columns={"ds": "date", "yhat": "forecast"})
    else:
        forecasts["ensemble"] = baseline_forecast.rename(columns={"ds": "date", "yhat": "forecast"})

    history_tail = series.tail(min(len(series), 30))
    inventory = inventory_recommendation(history_tail["target"])

    freq = infer_frequency(series["date"])
    for name, frame in forecasts.items():
        frame["model_name"] = name
        frame["freq"] = freq
        if "store_id" not in frame.columns:
            frame["store_id"] = series["store_id"].iloc[-1]
        if "item_id" not in frame.columns:
            frame["item_id"] = series["item_id"].iloc[-1]

    if output_dir is not None:
        build_powerbi_tables(series, forecasts, output_dir=Path(output_dir))

    metrics["forecast_models"] = {
        name: float(frame["forecast"].mean()) for name, frame in forecasts.items() if not frame.empty
    }

    return PipelineResult(
        raw=series,
        metrics=metrics,
        forecasts=forecasts,
        inventory=inventory,
    )


def make_forecast(source: str | Path, profile: str | None = None, horizon: int = 30) -> pd.DataFrame:
    df = load_and_prepare(source, profile=profile)
    series = aggregate_series(df)
    freq = infer_frequency(series["date"])
    try:
        _, forecast = try_train_prophet(series, horizon=horizon)
        forecast = forecast.rename(columns={"ds": "date", "yhat": "forecast"})
    except Exception:
        forecast = seasonal_naive_forecast(series, horizon=horizon).rename(columns={"ds": "date", "yhat": "forecast"})
    forecast["freq"] = freq
    forecast["store_id"] = series["store_id"].iloc[-1]
    forecast["item_id"] = series["item_id"].iloc[-1]
    return forecast


def maybe_train_lstm(source: str | Path, profile: str | None = None, horizon: int = 30):
    df = load_and_prepare(source, profile=profile)
    series = aggregate_series(df)
    return try_train_lstm(series, horizon=horizon)
