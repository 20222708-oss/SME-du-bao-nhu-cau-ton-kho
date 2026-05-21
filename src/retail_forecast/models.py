from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .features import prepare_supervised_frame

try:
    import pickle
except Exception:  # pragma: no cover
    pickle = None


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), 1e-6, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


@dataclass
class LinearModel:
    coef_: np.ndarray
    intercept_: float
    feature_columns: list[str]

    def predict(self, frame: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(frame, pd.DataFrame):
            X = frame[self.feature_columns].to_numpy(dtype=float)
        else:
            X = np.asarray(frame, dtype=float)
        return X @ self.coef_ + self.intercept_


def regression_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"date", "target", "store_id", "item_id"}
    return [c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]


def _fit_linear(X: np.ndarray, y: np.ndarray, alpha: float = 0.0) -> tuple[np.ndarray, float]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    ones = np.ones((X.shape[0], 1))
    Xb = np.hstack([ones, X])
    xtx = Xb.T @ Xb
    if alpha > 0:
        reg = np.eye(xtx.shape[0])
        reg[0, 0] = 0.0
        xtx = xtx + alpha * reg
    xty = Xb.T @ y
    beta = np.linalg.pinv(xtx) @ xty
    return beta[1:], float(beta[0])


def train_regression_model(df: pd.DataFrame, kind: str = "ridge") -> tuple[LinearModel, dict[str, float]]:
    frame = prepare_supervised_frame(df)
    feature_cols = regression_feature_columns(frame)
    X = frame[feature_cols].to_numpy(dtype=float)
    y = frame["target"].astype(float).to_numpy()

    split_idx = max(int(len(frame) * 0.8), 1)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    alpha = 1.0 if kind == "ridge" else 0.0
    coef, intercept = _fit_linear(X_train, y_train, alpha=alpha)
    model = LinearModel(coef_=coef, intercept_=intercept, feature_columns=feature_cols)

    X_eval = X_test if len(X_test) else X_train
    y_eval = y_test if len(y_test) else y_train
    pred = model.predict(X_eval)
    denom = pd.Series(y_eval).abs().clip(lower=1e-6).to_numpy()
    metrics = {
        "mae": float(np.mean(np.abs(y_eval - pred))),
        "rmse": float(np.sqrt(np.mean((y_eval - pred) ** 2))),
        "mape": mape(y_eval, pred),
        "bias": float(np.mean((pred - y_eval) / denom) * 100),
    }
    return model, metrics


def save_bundle(bundle: LinearModel, path: str) -> None:
    if pickle is None:
        raise RuntimeError("pickle is unavailable in this environment")
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_bundle(path: str) -> LinearModel:
    if pickle is None:
        raise RuntimeError("pickle is unavailable in this environment")
    with open(path, "rb") as f:
        return pickle.load(f)


def try_train_prophet(df: pd.DataFrame, horizon: int = 30):
    try:
        from prophet import Prophet
    except Exception as exc:  # pragma: no cover
        raise ImportError("prophet is not installed") from exc

    series = (
        df.groupby("date", as_index=False)["target"]
        .sum()
        .rename(columns={"date": "ds", "target": "y"})
        .sort_values("ds")
    )
    model = Prophet(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    model.fit(series)
    future = model.make_future_dataframe(periods=horizon, freq="D")
    forecast = model.predict(future)
    return model, forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def seasonal_naive_forecast(df: pd.DataFrame, horizon: int = 30, season_length: int = 7) -> pd.DataFrame:
    series = (
        df.groupby("date", as_index=False)["target"]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )
    if series.empty:
        raise ValueError("No observations available for forecasting.")

    history = series["target"].astype(float).to_list()
    last_date = pd.to_datetime(series["date"]).max()
    future_dates = pd.date_range(last_date, periods=horizon + 1, freq="D")[1:]
    preds = []
    for i in range(horizon):
        idx = len(history) - season_length + (i % season_length)
        if idx >= 0 and idx < len(history):
            preds.append(float(history[idx]))
        else:
            preds.append(float(history[-1]))
    return pd.DataFrame({"ds": future_dates, "yhat": preds})


def try_train_lstm(df: pd.DataFrame, horizon: int = 30, lookback: int = 30):
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras import Sequential
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Dense, Dropout, LSTM
        from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
    except Exception as exc:  # pragma: no cover
        raise ImportError("tensorflow is not installed") from exc

    series = (
        df.groupby("date", as_index=False)["target"]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )
    values = series["target"].astype("float32").values.reshape(-1, 1)

    if len(values) <= lookback + 5:
        raise ValueError("Not enough observations for LSTM training.")

    mean = float(values.mean())
    std = float(values.std()) or 1.0
    scaled = (values - mean) / std

    generator = TimeseriesGenerator(scaled, scaled, length=lookback, batch_size=16)

    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(generator, epochs=30, verbose=0, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

    preds = []
    window = scaled[-lookback:].reshape(1, lookback, 1)
    for _ in range(horizon):
        yhat = model.predict(window, verbose=0)[0, 0]
        preds.append(yhat)
        window = np.concatenate([window[:, 1:, :], np.array([[[yhat]]])], axis=1)

    preds = np.asarray(preds).reshape(-1, 1) * std + mean
    preds = preds.ravel()
    future_dates = pd.date_range(series["date"].max(), periods=horizon + 1, freq="D")[1:]
    forecast = pd.DataFrame({"ds": future_dates, "yhat": preds})
    return model, forecast


def combine_forecasts(*forecasts: pd.DataFrame) -> pd.DataFrame:
    valid = [f.copy() for f in forecasts if f is not None and not f.empty]
    if not valid:
        raise ValueError("At least one forecast is required.")

    merged = valid[0][["ds", "yhat"]].rename(columns={"yhat": "yhat_1"})
    for idx, frame in enumerate(valid[1:], start=2):
        merged = merged.merge(frame[["ds", "yhat"]].rename(columns={"yhat": f"yhat_{idx}"}), on="ds", how="outer")
    pred_cols = [c for c in merged.columns if c.startswith("yhat_")]
    merged["yhat"] = merged[pred_cols].mean(axis=1, skipna=True)
    merged = merged.sort_values("ds").reset_index(drop=True)
    return merged[["ds", "yhat"]]


def forecast_single_group(
    group_df: pd.DataFrame,
    horizon: int = 30,
    season_length: int = 7,
    enable_prophet: bool = True,
    enable_lstm: bool = True,
):
    """Forecast a single store/product time series.

    Returns a dict with baseline/prophet/lstm/ensemble DataFrames.
    Missing models are skipped gracefully.
    """

    group_df = group_df.sort_values("date").reset_index(drop=True)
    if group_df.empty:
        raise ValueError("Empty group series.")

    store_id = group_df["store_id"].iloc[0]
    item_id = group_df["item_id"].iloc[0]

    forecasts: dict[str, pd.DataFrame] = {}

    baseline = seasonal_naive_forecast(group_df, horizon=horizon, season_length=season_length)
    baseline["store_id"] = store_id
    baseline["item_id"] = item_id
    forecasts["baseline"] = baseline

    prophet_forecast = None
    if enable_prophet and len(group_df) >= max(30, season_length * 4):
        try:
            _, prophet_forecast = try_train_prophet(group_df, horizon=horizon)
            prophet_forecast["store_id"] = store_id
            prophet_forecast["item_id"] = item_id
            forecasts["prophet"] = prophet_forecast
        except Exception:
            prophet_forecast = None

    lstm_forecast = None
    if enable_lstm and len(group_df) >= 60:
        try:
            _, lstm_forecast = try_train_lstm(group_df, horizon=horizon)
            lstm_forecast["store_id"] = store_id
            lstm_forecast["item_id"] = item_id
            forecasts["lstm"] = lstm_forecast
        except Exception:
            lstm_forecast = None

    if prophet_forecast is not None and lstm_forecast is not None:
        ensemble = combine_forecasts(prophet_forecast, lstm_forecast)
        ensemble["store_id"] = store_id
        ensemble["item_id"] = item_id
        forecasts["ensemble"] = ensemble
    elif prophet_forecast is not None:
        forecasts["ensemble"] = prophet_forecast[["ds", "yhat", "store_id", "item_id"]].rename(
            columns={"ds": "date", "yhat": "forecast"}
        )
    elif lstm_forecast is not None:
        forecasts["ensemble"] = lstm_forecast[["ds", "yhat", "store_id", "item_id"]].rename(
            columns={"ds": "date", "yhat": "forecast"}
        )
    else:
        forecasts["ensemble"] = baseline.copy()

    for name, frame in list(forecasts.items()):
        if "ds" in frame.columns:
            frame = frame.rename(columns={"ds": "date", "yhat": "forecast"})
        frame["model_name"] = name
        forecasts[name] = frame[["date", "forecast", "model_name", "store_id", "item_id"] + [c for c in frame.columns if c not in {"date", "forecast", "model_name", "store_id", "item_id"}]]

    return forecasts


def forecast_many_groups(
    df: pd.DataFrame,
    horizon: int = 30,
    group_cols: tuple[str, str] = ("store_id", "item_id"),
    max_groups: int | None = None,
    enable_prophet: bool = True,
    enable_lstm: bool = True,
):
    """Run forecasts for many store/product groups and combine by model name."""

    groups = []
    for keys, group in df.groupby(list(group_cols), sort=False):
        if isinstance(keys, tuple):
            store_id, item_id = keys
        else:
            store_id, item_id = keys, None
        groups.append((store_id, item_id, group))

    if max_groups is not None:
        groups = sorted(groups, key=lambda item: float(item[2]["target"].sum()), reverse=True)[:max_groups]

    frames: dict[str, list[pd.DataFrame]] = {"baseline": [], "prophet": [], "lstm": [], "ensemble": []}
    for store_id, item_id, group in groups:
        group_forecasts = forecast_single_group(
            group,
            horizon=horizon,
            enable_prophet=enable_prophet,
            enable_lstm=enable_lstm,
        )
        for name, frame in group_forecasts.items():
            frames.setdefault(name, []).append(frame)

    combined: dict[str, pd.DataFrame] = {}
    for name, parts in frames.items():
        if parts:
            combined[name] = pd.concat(parts, ignore_index=True)
    return combined
