from __future__ import annotations

import pandas as pd


def make_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dayofweek"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["quarter"] = out["date"].dt.quarter
    out["day"] = out["date"].dt.day
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype(int)
    out["year"] = out["date"].dt.year
    out["is_month_start"] = out["date"].dt.is_month_start.astype(int)
    out["is_month_end"] = out["date"].dt.is_month_end.astype(int)
    return out


def make_lag_features(
    df: pd.DataFrame,
    target_col: str = "target",
    lags: tuple[int, ...] = (1, 7, 14, 28),
    windows: tuple[int, ...] = (7, 14, 28),
) -> pd.DataFrame:
    out = df.copy().sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    def _per_group(group: pd.DataFrame) -> pd.DataFrame:
        g = group.sort_values("date").copy()
        for lag in lags:
            g[f"lag_{lag}"] = g[target_col].shift(lag)
        shifted = g[target_col].shift(1)
        for window in windows:
            g[f"roll_mean_{window}"] = shifted.rolling(window=window, min_periods=1).mean()
            g[f"roll_std_{window}"] = shifted.rolling(window=window, min_periods=2).std()
        return g

    out = out.groupby(["store_id", "item_id"], group_keys=False).apply(_per_group)
    return out.reset_index(drop=True)


def prepare_supervised_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["promo", "holiday", "transactions", "price", "oil", "cpi", "unemployment"]:
        if col not in out.columns:
            out[col] = 0
    out = make_time_features(out)
    out = make_lag_features(out)

    numeric_cols = ["promo", "holiday", "transactions", "price", "oil", "cpi", "unemployment"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["promo"] = out["promo"].fillna(0).astype(float)
    out["holiday"] = out["holiday"].fillna(0).astype(float)

    out = out.dropna(subset=["lag_1", "lag_7", "roll_mean_7"])
    out = out.fillna(0)
    return out


def build_forecast_frame(
    history: pd.DataFrame,
    horizon: int,
    freq: str = "D",
    exog: dict[str, object] | None = None,
) -> pd.DataFrame:
    exog = exog or {}
    last_date = pd.to_datetime(history["date"]).max()
    future_dates = pd.date_range(last_date, periods=horizon + 1, freq=freq)[1:]
    future = pd.DataFrame({"date": future_dates})
    future["store_id"] = history["store_id"].iloc[-1]
    future["item_id"] = history["item_id"].iloc[-1]
    for key, value in exog.items():
        future[key] = value
    return future
