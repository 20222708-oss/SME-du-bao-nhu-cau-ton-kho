from __future__ import annotations

from pathlib import Path

import pandas as pd

from .inventory import inventory_recommendation


def build_powerbi_tables(
    history: pd.DataFrame,
    forecast_tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Build BI-friendly CSV tables.

    Returns paths for:
    - fact_history.csv
    - fact_forecast.csv
    - dim_product.csv
    - dim_store.csv
    - inventory_recommendations.csv
    - model_metrics.csv
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fact_history = history.copy()
    fact_history = fact_history.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    fact_history["table_type"] = "history"
    fact_history_path = output_dir / "fact_history.csv"
    fact_history.to_csv(fact_history_path, index=False)

    fact_forecasts = []
    for model_name, frame in forecast_tables.items():
        if frame is None or frame.empty:
            continue
        tmp = frame.copy()
        tmp["model_name"] = model_name
        tmp["table_type"] = "forecast"
        if "yhat" in tmp.columns and "forecast" not in tmp.columns:
            tmp = tmp.rename(columns={"yhat": "forecast"})
        fact_forecasts.append(tmp)
    fact_forecast = pd.concat(fact_forecasts, ignore_index=True) if fact_forecasts else pd.DataFrame()
    fact_forecast_path = output_dir / "fact_forecast.csv"
    fact_forecast.to_csv(fact_forecast_path, index=False)

    product_cols = [c for c in ["item_id", "product_name", "category", "base_price", "base_demand"] if c in history.columns]
    dim_product = history[product_cols].drop_duplicates(subset=["item_id"]).sort_values("item_id")
    dim_product_path = output_dir / "dim_product.csv"
    dim_product.to_csv(dim_product_path, index=False)

    store_cols = [c for c in ["store_id", "store_name", "city", "district", "store_type"] if c in history.columns]
    dim_store = history[store_cols].drop_duplicates(subset=["store_id"]).sort_values("store_id")
    dim_store_path = output_dir / "dim_store.csv"
    dim_store.to_csv(dim_store_path, index=False)

    inventory_rows = []
    if not fact_forecast.empty:
        preferred_order = ["ensemble", "prophet", "lstm", "baseline"]
        for (store_id, item_id), group in fact_forecast.groupby(["store_id", "item_id"], dropna=False):
            model_name = next((name for name in preferred_order if name in set(group["model_name"])), group["model_name"].iloc[0])
            selected = group[group["model_name"] == model_name].sort_values("date")
            rec = inventory_recommendation(selected["forecast"].tail(min(len(selected), 30)))
            item_meta = {}
            store_meta = {}
            if not dim_product.empty:
                prod = dim_product[dim_product["item_id"] == item_id]
                if not prod.empty:
                    item_meta = prod.iloc[0].to_dict()
            if not dim_store.empty:
                store = dim_store[dim_store["store_id"] == store_id]
                if not store.empty:
                    store_meta = store.iloc[0].to_dict()
            inventory_rows.append({"store_id": store_id, "item_id": item_id, "source_model": model_name, **item_meta, **store_meta, **rec})
    inventory_df = pd.DataFrame(inventory_rows)
    inventory_path = output_dir / "inventory_recommendations.csv"
    inventory_df.to_csv(inventory_path, index=False)

    metrics_path = output_dir / "model_metrics.csv"
    metrics_df = pd.DataFrame(
        [
            {
                "model_name": name,
                "rows": len(frame) if frame is not None else 0,
                "min_forecast_date": frame["date"].min() if frame is not None and not frame.empty else None,
                "max_forecast_date": frame["date"].max() if frame is not None and not frame.empty else None,
                "groups": frame[["store_id", "item_id"]].drop_duplicates().shape[0] if frame is not None and not frame.empty and {"store_id", "item_id"} <= set(frame.columns) else 0,
            }
            for name, frame in forecast_tables.items()
        ]
    )
    metrics_df.to_csv(metrics_path, index=False)

    return {
        "fact_history": fact_history_path,
        "fact_forecast": fact_forecast_path,
        "dim_product": dim_product_path,
        "dim_store": dim_store_path,
        "inventory_recommendations": inventory_path,
        "model_metrics": metrics_path,
    }
