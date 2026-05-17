from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


CANONICAL_COLUMNS = {
    "date": "date",
    "ds": "date",
    "day": "date",
    "datetime": "date",
    "store": "store_id",
    "store_id": "store_id",
    "store_nbr": "store_id",
    "store number": "store_id",
    "item": "item_id",
    "item_id": "item_id",
    "family": "item_id",
    "department": "item_id",
    "dept": "item_id",
    "product": "item_id",
    "product_id": "item_id",
    "sales": "target",
    "weekly_sales": "target",
    "weekly sales": "target",
    "unit_sales": "target",
    "qty_sold": "target",
    "quantity_sold": "target",
    "sold_qty": "target",
    "quantity": "target",
    "y": "target",
    "promo": "promo",
    "promotion": "promo",
    "onpromotion": "promo",
    "isholiday": "holiday",
    "holiday": "holiday",
    "transactions": "transactions",
    "price": "price",
    "sell_price": "price",
    "oil": "oil",
    "cpi": "cpi",
    "unemployment": "unemployment",
}

ESSENTIAL_COLUMNS = {
    "date",
    "ds",
    "day",
    "datetime",
    "store",
    "store_id",
    "store_nbr",
    "item",
    "item_id",
    "family",
    "department",
    "product",
    "product_id",
    "sales",
    "weekly_sales",
    "unit_sales",
    "qty_sold",
    "quantity_sold",
    "sold_qty",
    "quantity",
    "y",
    "promo",
    "promotion",
    "onpromotion",
    "isholiday",
    "holiday",
    "transactions",
    "price",
    "sell_price",
    "oil",
    "cpi",
    "unemployment",
    "stock_begin",
    "stock_end",
    "lead_time_days",
    "revenue",
    "discount_pct",
    "is_weekend",
    "is_holiday",
    "holiday_name",
    "is_tet_season",
    "day_of_week",
    "month",
    "quarter",
    "year",
    "product_name",
    "category",
    "store_name",
    "city",
    "district",
    "store_type",
    "stock_begin",
    "stock_end",
    "lead_time_days",
    "target",
}


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    date_candidates: tuple[str, ...]
    target_candidates: tuple[str, ...]
    store_candidates: tuple[str, ...] = ("store_id", "store", "store_nbr")
    item_candidates: tuple[str, ...] = ("item_id", "item", "family", "department", "product", "product_id")


PROFILES = {
    "auto": DatasetProfile(name="auto", date_candidates=("date",), target_candidates=("target", "sales")),
    "m5": DatasetProfile(name="m5", date_candidates=("date",), target_candidates=("sales", "unit_sales")),
    "rossmann": DatasetProfile(name="rossmann", date_candidates=("date",), target_candidates=("sales",)),
    "walmart": DatasetProfile(name="walmart", date_candidates=("date",), target_candidates=("weekly_sales", "sales")),
    "favorita": DatasetProfile(name="favorita", date_candidates=("date",), target_candidates=("sales", "unit_sales")),
    "store_sales": DatasetProfile(name="store_sales", date_candidates=("date",), target_candidates=("sales", "quantity", "sum_total")),
}


def _lower_map(columns: Iterable[str]) -> dict[str, str]:
    return {c.lower().strip(): c for c in columns}


def guess_profile(columns: Iterable[str]) -> str:
    cols = {c.lower().strip() for c in columns}
    if any(re.match(r"^d_\d+$", c) for c in cols):
        return "m5"
    if {"store_nbr", "family"} <= cols:
        return "favorita"
    if {"store_nbr", "transactions"} <= cols:
        return "store_sales"
    if {"weekly_sales", "isholiday"} <= cols:
        return "walmart"
    if {"store", "sales"} <= cols and "promo" in cols:
        return "rossmann"
    return "auto"


def standardize_dataframe(df: pd.DataFrame, profile: str | None = None) -> pd.DataFrame:
    df = df.copy()
    lower_lookup = _lower_map(df.columns)
    detected_profile = profile or guess_profile(df.columns)
    profile_cfg = PROFILES.get(detected_profile, PROFILES["auto"])

    rename_map: dict[str, str] = {}
    for raw_lower, canonical in CANONICAL_COLUMNS.items():
        if raw_lower in lower_lookup:
            rename_map[lower_lookup[raw_lower]] = canonical
    df = df.rename(columns=rename_map)

    if "date" not in df.columns:
        raise ValueError("Could not find a date column in the dataset.")
    if "target" not in df.columns:
        raise ValueError("Could not find a target/sales column in the dataset.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if "store_id" not in df.columns:
        df["store_id"] = "store_0"
    if "item_id" not in df.columns:
        df["item_id"] = "item_0"

    for col in ["promo", "holiday", "transactions", "price", "oil", "cpi", "unemployment"]:
        if col not in df.columns:
            df[col] = pd.NA

    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    df.attrs["profile"] = detected_profile
    df.attrs["profile_name"] = profile_cfg.name
    return df


def load_tabular(path: str | Path, profile: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        try:
            df = pd.read_csv(path, low_memory=False, usecols=lambda c: c.lower().strip() in ESSENTIAL_COLUMNS)
        except ValueError:
            df = pd.read_csv(path, low_memory=False)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return standardize_dataframe(df, profile=profile)


def _find_first_file(folder: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = folder / name
        if candidate.exists():
            return candidate
    return None


def _load_m5_folder(folder: Path) -> pd.DataFrame:
    sales_file = _find_first_file(folder, ["sales_train_validation.csv", "sales_train_evaluation.csv"])
    if sales_file is None:
        raise FileNotFoundError("M5 folder must contain sales_train_validation.csv or sales_train_evaluation.csv")

    sales = pd.read_csv(sales_file)
    id_col = "id" if "id" in sales.columns else sales.columns[0]
    value_cols = [c for c in sales.columns if re.match(r"^d_\d+$", str(c).strip().lower())]
    if not value_cols:
        raise ValueError("M5 sales file does not contain daily columns d_1, d_2, ...")

    long_df = sales.melt(id_vars=[id_col], value_vars=value_cols, var_name="d", value_name="target")
    ids = long_df[id_col].astype(str)
    parsed = ids.str.extract(r"^(?P<item>.+)_(?P<store>[A-Z]{2}_\d+)_(?:evaluation|validation)$")
    long_df["item_id"] = parsed["item"].fillna(ids)
    long_df["store_id"] = parsed["store"].fillna("store_0")

    calendar_file = _find_first_file(folder, ["calendar.csv"])
    if calendar_file is not None:
        calendar = pd.read_csv(calendar_file)
        rename_map = {}
        if "date" in calendar.columns:
            rename_map["date"] = "date"
        if "d" in calendar.columns:
            rename_map["d"] = "d"
        calendar = calendar.rename(columns=rename_map)
        if "date" in calendar.columns and "d" in calendar.columns:
            calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
            keep_cols = ["d", "date"]
            for extra in ["event_name_1", "event_name_2", "snap_CA", "snap_TX", "snap_WI", "weekday", "wm_yr_wk"]:
                if extra in calendar.columns:
                    keep_cols.append(extra)
            long_df = long_df.merge(calendar[keep_cols], on="d", how="left")
        else:
            long_df["date"] = pd.date_range("2011-01-01", periods=len(long_df["d"].unique()), freq="D").repeat(
                sales.shape[0]
            )
    else:
        unique_days = sorted(long_df["d"].unique(), key=lambda x: int(str(x).split("_")[1]))
        day_map = {day: idx for idx, day in enumerate(unique_days)}
        long_df["date"] = pd.Timestamp("2011-01-01") + pd.to_timedelta(long_df["d"].map(day_map), unit="D")

    for col in ["promo", "holiday", "transactions", "price", "oil", "cpi", "unemployment"]:
        if col not in long_df.columns:
            long_df[col] = pd.NA

    return standardize_dataframe(long_df[["date", "store_id", "item_id", "target"] + [c for c in long_df.columns if c in {"promo", "holiday", "transactions", "price", "oil", "cpi", "unemployment"}]], profile="m5")


def _load_generic_folder(folder: Path) -> pd.DataFrame:
    for name in ["train.csv", "sales.csv", "data.csv"]:
        file_path = folder / name
        if file_path.exists():
            return load_tabular(file_path)
    raise FileNotFoundError("Could not find a supported CSV file in the folder.")


def load_dataset(source: str | Path, profile: str | None = None) -> pd.DataFrame:
    path = Path(source)
    if path.is_dir():
        profile = profile or guess_profile([p.name for p in path.iterdir() if p.is_file()])
        if profile == "m5":
            return _load_m5_folder(path)
        return _load_generic_folder(path)
    return load_tabular(path, profile=profile)


def infer_frequency(date_series: pd.Series) -> str:
    dates = pd.to_datetime(date_series).dropna().sort_values()
    if len(dates) < 3:
        return "D"
    delta = dates.diff().dropna().dt.days.median()
    if pd.isna(delta):
        return "D"
    if delta >= 27:
        return "M"
    if delta >= 6:
        return "W"
    return "D"


def aggregate_series(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    group_cols = group_cols or ["date"]
    if group_cols == ["date"]:
        series = (
            df.groupby("date", as_index=False)["target"]
            .sum()
            .sort_values("date")
            .reset_index(drop=True)
        )
        series["store_id"] = "all"
        series["item_id"] = "all"
        return series
    return (
        df.groupby(group_cols + ["date"], as_index=False)["target"]
        .sum()
        .sort_values(group_cols + ["date"])
        .reset_index(drop=True)
    )
