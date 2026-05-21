from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from .datasets import load_dataset
from .models import seasonal_naive_forecast


ROOT_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = ROOT_DIR / "web"


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    for column in frame.columns:
        lower = column.lower().strip()
        if lower in {
            "date",
            "ds",
            "date_start",
            "date_end",
            "order_date",
            "expected_delivery_date",
            "actual_delivery_date",
            "opening_date",
            "min_forecast_date",
            "max_forecast_date",
        }:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].dt.strftime("%Y-%m-%d")
    out = out.where(pd.notna(out), None)
    return out.to_dict(orient="records")


def _compact_number(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "0"
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f} tỷ".replace(".", ",")
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f} tr".replace(".", ",")
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f} nghìn".replace(".", ",")
    if value.is_integer():
        return f"{int(value):,}".replace(",", ".")
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _first_existing(roots: list[Path], names: list[str]) -> Path | None:
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.exists():
                return candidate
    return None


def _filter_frame(
    frame: pd.DataFrame,
    *,
    store_id: str | None = None,
    item_id: str | None = None,
    category: str | None = None,
    model_name: str | None = None,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if store_id and store_id != "all" and "store_id" in out.columns:
        out = out[out["store_id"].astype(str) == str(store_id)]
    if item_id and item_id != "all" and "item_id" in out.columns:
        out = out[out["item_id"].astype(str) == str(item_id)]
    if category and category != "all" and "category" in out.columns:
        out = out[out["category"].astype(str) == str(category)]
    if model_name and model_name != "all" and "model_name" in out.columns:
        out = out[out["model_name"].astype(str) == str(model_name)]
    return out.reset_index(drop=True)


def _limit_history_window(frame: pd.DataFrame, days: int = 120) -> pd.DataFrame:
    if frame is None or frame.empty or "date" not in frame.columns:
        return pd.DataFrame()
    out = frame.copy().sort_values("date")
    cutoff = pd.to_datetime(out["date"]).max() - pd.Timedelta(days=days)
    return out[pd.to_datetime(out["date"]) >= cutoff].reset_index(drop=True)


def _ensure_category_columns(frame: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if products is None or products.empty or "item_id" not in out.columns:
        return out
    dim = products.copy()
    if "item_id" not in dim.columns:
        return out
    for column in ["product_name", "category", "sub_category", "brand"]:
        if column not in out.columns and column in dim.columns:
            out = out.merge(dim[["item_id", column]].drop_duplicates("item_id"), on="item_id", how="left")
    return out


def _ensure_store_columns(frame: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if stores is None or stores.empty or "store_id" not in out.columns:
        return out
    dim = stores.copy()
    if "store_id" not in dim.columns:
        return out
    for column in ["store_name", "city", "district", "region", "store_type"]:
        if column not in out.columns and column in dim.columns:
            out = out.merge(dim[["store_id", column]].drop_duplicates("store_id"), on="store_id", how="left")
    return out


@dataclass
class DashboardRepository:
    data_root: Path
    artifacts_root: Path | None = None

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root).expanduser().resolve()
        self.artifacts_root = Path(self.artifacts_root).expanduser().resolve() if self.artifacts_root else None
        self.roots: list[Path] = [root for root in [self.artifacts_root, self.data_root] if root is not None]

    @cached_property
    def history(self) -> pd.DataFrame:
        for root in self.roots:
            try:
                if root.is_dir():
                    frame = load_dataset(root, profile="auto")
                else:
                    frame = load_dataset(root, profile="auto")
                if not frame.empty:
                    frame = _ensure_category_columns(frame, self.products)
                    frame = _ensure_store_columns(frame, self.stores)
                    if "date" in frame.columns:
                        frame = frame.sort_values("date").reset_index(drop=True)
                    return frame
            except Exception:
                continue
        raise FileNotFoundError(
            "Could not find a compatible history table. Expected fact_history.csv, fact_sales.csv, train.csv, sales.csv or data.csv."
        )

    @cached_property
    def forecast(self) -> pd.DataFrame:
        file_path = _first_existing(self.roots, ["fact_forecast.csv", "forecast.csv"])
        if file_path is None:
            return pd.DataFrame()
        frame = _read_csv(file_path)
        rename_map = {}
        if "ds" in frame.columns and "date" not in frame.columns:
            rename_map["ds"] = "date"
        if "yhat" in frame.columns and "forecast" not in frame.columns:
            rename_map["yhat"] = "forecast"
        frame = frame.rename(columns=rename_map)
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = _ensure_category_columns(frame, self.products)
        frame = _ensure_store_columns(frame, self.stores)
        frame = frame.dropna(subset=["date"]) if "date" in frame.columns else frame
        if "model_name" not in frame.columns:
            frame["model_name"] = "baseline"
        return frame.sort_values(["model_name", "store_id", "item_id", "date"]).reset_index(drop=True)

    @cached_property
    def products(self) -> pd.DataFrame:
        file_path = _first_existing(self.roots, ["dim_product.csv"])
        if file_path is None:
            return pd.DataFrame()
        frame = _read_csv(file_path)
        return frame

    @cached_property
    def stores(self) -> pd.DataFrame:
        file_path = _first_existing(self.roots, ["dim_store.csv"])
        if file_path is None:
            return pd.DataFrame()
        frame = _read_csv(file_path)
        return frame

    @cached_property
    def inventory(self) -> pd.DataFrame:
        file_path = _first_existing(self.roots, ["inventory_recommendations.csv"])
        if file_path is None:
            return pd.DataFrame()
        frame = _read_csv(file_path)
        frame = _ensure_category_columns(frame, self.products)
        frame = _ensure_store_columns(frame, self.stores)
        return frame

    @cached_property
    def metrics(self) -> pd.DataFrame:
        file_path = _first_existing(self.roots, ["model_metrics.csv"])
        if file_path is None:
            json_path = _first_existing(self.roots, ["metrics.json"])
            if json_path is None:
                return pd.DataFrame()
            try:
                return pd.read_json(json_path, orient="records")
            except Exception:
                return pd.DataFrame()
        frame = _read_csv(file_path)
        return frame

    def options(self) -> dict[str, list[dict[str, str]]]:
        stores = self.stores
        products = self.products
        history = self.history

        if stores.empty and "store_id" in history.columns:
            store_values = sorted(history["store_id"].dropna().astype(str).unique().tolist())
            stores = pd.DataFrame({"store_id": store_values})
        if products.empty and "item_id" in history.columns:
            product_values = sorted(history["item_id"].dropna().astype(str).unique().tolist())
            products = pd.DataFrame({"item_id": product_values})

        if not stores.empty and "store_name" not in stores.columns:
            stores["store_name"] = stores["store_id"].astype(str)
        if not products.empty and "product_name" not in products.columns:
            products["product_name"] = products["item_id"].astype(str)

        store_options = []
        for _, row in stores.sort_values("store_id").drop_duplicates("store_id").iterrows():
            label = str(row.get("store_name") or row.get("store_id"))
            if row.get("city"):
                label = f"{label} · {row.get('city')}"
            store_options.append({"value": str(row.get("store_id")), "label": label})

        product_options = []
        for _, row in products.sort_values("item_id").drop_duplicates("item_id").iterrows():
            label = str(row.get("product_name") or row.get("item_id"))
            if row.get("category"):
                label = f"{label} · {row.get('category')}"
            product_options.append({"value": str(row.get("item_id")), "label": label})

        categories = []
        if "category" in products.columns:
            categories = sorted(products["category"].dropna().astype(str).unique().tolist())
        elif "category" in history.columns:
            categories = sorted(history["category"].dropna().astype(str).unique().tolist())

        models = []
        if not self.forecast.empty and "model_name" in self.forecast.columns:
            models = sorted(self.forecast["model_name"].dropna().astype(str).unique().tolist())
        if not models:
            models = ["ensemble", "baseline"]

        return {
            "stores": store_options,
            "products": product_options,
            "categories": [{"value": value, "label": value} for value in categories],
            "models": [{"value": value, "label": value.capitalize()} for value in models],
        }

    def _forecast_for_scope(
        self,
        *,
        store_id: str | None,
        item_id: str | None,
        category: str | None,
        model_name: str,
        horizon: int,
    ) -> pd.DataFrame:
        selected = _filter_frame(
            self.forecast,
            store_id=store_id,
            item_id=item_id,
            category=category,
            model_name=model_name,
        )
        if not selected.empty and "forecast" in selected.columns:
            selected = selected.copy()
            selected["date"] = pd.to_datetime(selected["date"], errors="coerce")
            selected = selected.dropna(subset=["date"])
            selected = selected[selected["date"] > pd.to_datetime(self.history["date"]).max()]
            if not selected.empty:
                return selected.groupby("date", as_index=False)["forecast"].sum().sort_values("date").reset_index(drop=True)

        scope_history = _filter_frame(self.history, store_id=store_id, item_id=item_id, category=category)
        if scope_history.empty:
            return pd.DataFrame(columns=["date", "forecast"])
        naive = seasonal_naive_forecast(scope_history[["date", "target"]].copy(), horizon=horizon)
        naive = naive.rename(columns={"ds": "date", "yhat": "forecast"})
        naive["date"] = pd.to_datetime(naive["date"], errors="coerce")
        return naive.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    def series(
        self,
        *,
        store_id: str | None,
        item_id: str | None,
        category: str | None,
        model_name: str,
        horizon: int,
        history_window: int,
    ) -> dict[str, list[dict[str, Any]]]:
        history = _filter_frame(self.history, store_id=store_id, item_id=item_id, category=category)
        actual = (
            history.groupby("date", as_index=False)["target"]
            .sum()
            .sort_values("date")
            .tail(max(history_window, 30))
            .reset_index(drop=True)
        )

        forecast = self._forecast_for_scope(
            store_id=store_id,
            item_id=item_id,
            category=category,
            model_name=model_name,
            horizon=horizon,
        )

        return {
            "actual": _records(actual.rename(columns={"target": "value"})),
            "forecast": _records(forecast.rename(columns={"forecast": "value"})),
        }

    def categories(
        self,
        *,
        store_id: str | None,
        item_id: str | None,
        category: str | None,
    ) -> list[dict[str, Any]]:
        history = _filter_frame(self.history, store_id=store_id, item_id=item_id, category=category)
        if history.empty or "category" not in history.columns:
            return []
        grouped = (
            history.groupby("category", as_index=False)["target"]
            .sum()
            .sort_values("target", ascending=False)
            .head(6)
            .reset_index(drop=True)
        )
        total = float(grouped["target"].sum()) or 1.0
        grouped["share"] = grouped["target"] / total * 100
        grouped = grouped.rename(columns={"target": "value", "category": "label"})
        return _records(grouped)

    def top_products(
        self,
        *,
        store_id: str | None,
        item_id: str | None,
        category: str | None,
        model_name: str,
    ) -> list[dict[str, Any]]:
        history = _filter_frame(self.history, store_id=store_id, item_id=item_id, category=category)
        if history.empty:
            return []
        cols = [c for c in ["item_id", "product_name", "category"] if c in history.columns]
        grouped = history.groupby(cols, as_index=False)["target"].sum().rename(columns={"target": "actual_units"})

        forecast = _filter_frame(self.forecast, store_id=store_id, item_id=item_id, category=category, model_name=model_name)
        if not forecast.empty and "forecast" in forecast.columns:
            fcols = [c for c in ["item_id", "product_name", "category"] if c in forecast.columns]
            forecast_grouped = (
                forecast.groupby(fcols, as_index=False)["forecast"].sum().rename(columns={"forecast": "forecast_units"})
            )
            merge_keys = [c for c in ["item_id", "product_name", "category"] if c in grouped.columns and c in forecast_grouped.columns]
            if merge_keys:
                grouped = grouped.merge(forecast_grouped, on=merge_keys, how="left")
            else:
                grouped["forecast_units"] = pd.NA
        else:
            grouped["forecast_units"] = pd.NA

        grouped = grouped.sort_values("actual_units", ascending=False).head(10).reset_index(drop=True)
        return _records(grouped)

    def inventory_table(
        self,
        *,
        store_id: str | None,
        item_id: str | None,
        category: str | None,
        model_name: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        inventory = _filter_frame(self.inventory, store_id=store_id, item_id=item_id, category=category)
        if inventory.empty:
            return []
        if "source_model" in inventory.columns and model_name and model_name != "all":
            scoped = inventory[inventory["source_model"].astype(str) == str(model_name)]
            if not scoped.empty:
                inventory = scoped
        sort_column = "reorder_point" if "reorder_point" in inventory.columns else "safety_stock"
        inventory = inventory.sort_values(sort_column, ascending=False).head(limit).reset_index(drop=True)
        wanted = [
            c
            for c in [
                "store_name",
                "city",
                "store_type",
                "item_id",
                "product_name",
                "category",
                "avg_demand",
                "demand_std",
                "safety_stock",
                "reorder_point",
                "eoq",
                "source_model",
            ]
            if c in inventory.columns
        ]
        return _records(inventory[wanted])

    def metrics_table(self) -> list[dict[str, Any]]:
        if self.metrics.empty:
            return []
        return _records(self.metrics)

    def summary(
        self,
        *,
        store_id: str | None,
        item_id: str | None,
        category: str | None,
        model_name: str,
        horizon: int,
    ) -> dict[str, Any]:
        history = _filter_frame(self.history, store_id=store_id, item_id=item_id, category=category)
        forecast = self._forecast_for_scope(
            store_id=store_id,
            item_id=item_id,
            category=category,
            model_name=model_name,
            horizon=horizon,
        )
        inventory = _filter_frame(self.inventory, store_id=store_id, item_id=item_id, category=category)

        total_units = float(history["target"].sum()) if "target" in history.columns and not history.empty else 0.0
        total_revenue = float(history["revenue"].sum()) if "revenue" in history.columns and not history.empty else 0.0
        forecast_units = float(forecast["forecast"].sum()) if "forecast" in forecast.columns and not forecast.empty else 0.0
        product_count = int(history["item_id"].nunique()) if "item_id" in history.columns and not history.empty else 0
        store_count = int(history["store_id"].nunique()) if "store_id" in history.columns and not history.empty else 0
        category_count = int(history["category"].nunique()) if "category" in history.columns and not history.empty else 0
        stockout_rate = float(history["stockout_flag"].mean() * 100) if "stockout_flag" in history.columns and not history.empty else 0.0
        safety_stock_avg = float(inventory["safety_stock"].mean()) if "safety_stock" in inventory.columns and not inventory.empty else 0.0
        reorder_point_avg = float(inventory["reorder_point"].mean()) if "reorder_point" in inventory.columns and not inventory.empty else 0.0
        eoq_avg = float(inventory["eoq"].mean()) if "eoq" in inventory.columns and not inventory.empty else 0.0
        coverage = 0.0
        if not self.metrics.empty and {"model_name", "groups"} <= set(self.metrics.columns):
            row = self.metrics[self.metrics["model_name"].astype(str) == str(model_name)]
            if row.empty and not self.metrics.empty:
                row = self.metrics.head(1)
            if not row.empty:
                coverage = float(row["groups"].iloc[0])

        first_history_date = pd.to_datetime(history["date"]).min() if not history.empty and "date" in history.columns else pd.NaT
        last_history_date = pd.to_datetime(history["date"]).max() if not history.empty and "date" in history.columns else pd.NaT
        last_forecast_date = pd.to_datetime(forecast["date"]).max() if not forecast.empty and "date" in forecast.columns else pd.NaT

        recent_window = _limit_history_window(history, 30)
        recent_units = float(recent_window["target"].sum()) if "target" in recent_window.columns and not recent_window.empty else 0.0
        forecast_rate = (forecast_units / recent_units * 100) if recent_units else 0.0

        return {
            "total_units": total_units,
            "total_revenue": total_revenue,
            "forecast_units": forecast_units,
            "product_count": product_count,
            "store_count": store_count,
            "category_count": category_count,
            "stockout_rate": stockout_rate,
            "safety_stock_avg": safety_stock_avg,
            "reorder_point_avg": reorder_point_avg,
            "eoq_avg": eoq_avg,
            "coverage": coverage,
            "forecast_rate": forecast_rate,
            "actual_start": first_history_date.strftime("%Y-%m-%d") if pd.notna(first_history_date) else None,
            "actual_end": last_history_date.strftime("%Y-%m-%d") if pd.notna(last_history_date) else None,
            "forecast_end": last_forecast_date.strftime("%Y-%m-%d") if pd.notna(last_forecast_date) else None,
            "model_name": model_name,
        }

    def dashboard(
        self,
        *,
        store_id: str | None,
        item_id: str | None,
        category: str | None,
        model_name: str,
        horizon: int = 30,
        history_window: int = 120,
        inventory_limit: int = 12,
    ) -> dict[str, Any]:
        return {
            "summary": self.summary(
                store_id=store_id,
                item_id=item_id,
                category=category,
                model_name=model_name,
                horizon=horizon,
            ),
            "series": self.series(
                store_id=store_id,
                item_id=item_id,
                category=category,
                model_name=model_name,
                horizon=horizon,
                history_window=history_window,
            ),
            "categories": self.categories(store_id=store_id, item_id=item_id, category=category),
            "top_products": self.top_products(
                store_id=store_id,
                item_id=item_id,
                category=category,
                model_name=model_name,
            ),
            "inventory": self.inventory_table(
                store_id=store_id,
                item_id=item_id,
                category=category,
                model_name=model_name,
                limit=inventory_limit,
            ),
            "metrics": self.metrics_table(),
            "filters": {
                "store_id": store_id or "all",
                "item_id": item_id or "all",
                "category": category or "all",
                "model_name": model_name,
                "horizon": horizon,
                "history_window": history_window,
            },
        }

    def bootstrap(
        self,
        *,
        store_id: str | None,
        item_id: str | None,
        category: str | None,
        model_name: str,
        horizon: int = 30,
        history_window: int = 120,
        inventory_limit: int = 12,
    ) -> dict[str, Any]:
        return {
            "options": self.options(),
            "dashboard": self.dashboard(
                store_id=store_id,
                item_id=item_id,
                category=category,
                model_name=model_name,
                horizon=horizon,
                history_window=history_window,
                inventory_limit=inventory_limit,
            ),
            "data_root": str(self.data_root),
            "artifacts_root": str(self.artifacts_root) if self.artifacts_root else None,
        }


def create_app(
    data_root: str | Path,
    artifacts_root: str | Path | None = None,
    *,
    title: str = "Retail Forecast Studio",
):
    from fastapi import FastAPI, Query
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    repo = DashboardRepository(Path(data_root), Path(artifacts_root) if artifacts_root else None)

    app = FastAPI(title=title)
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/")
    def index():
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            return JSONResponse({"error": "Frontend assets are missing."}, status_code=500)
        return FileResponse(index_path)

    @app.get("/api/bootstrap")
    def bootstrap(
        store_id: str = Query("all"),
        item_id: str = Query("all"),
        category: str = Query("all"),
        model_name: str = Query("ensemble"),
        horizon: int = Query(30, ge=1, le=365),
        history_window: int = Query(120, ge=30, le=365),
        inventory_limit: int = Query(12, ge=1, le=100),
    ):
        return repo.bootstrap(
            store_id=None if store_id == "all" else store_id,
            item_id=None if item_id == "all" else item_id,
            category=None if category == "all" else category,
            model_name=model_name,
            horizon=horizon,
            history_window=history_window,
            inventory_limit=inventory_limit,
        )

    @app.get("/api/dashboard")
    def dashboard(
        store_id: str = Query("all"),
        item_id: str = Query("all"),
        category: str = Query("all"),
        model_name: str = Query("ensemble"),
        horizon: int = Query(30, ge=1, le=365),
        history_window: int = Query(120, ge=30, le=365),
        inventory_limit: int = Query(12, ge=1, le=100),
    ):
        return repo.dashboard(
            store_id=None if store_id == "all" else store_id,
            item_id=None if item_id == "all" else item_id,
            category=None if category == "all" else category,
            model_name=model_name,
            horizon=horizon,
            history_window=history_window,
            inventory_limit=inventory_limit,
        )

    return app


def run_web_server(
    *,
    data_root: str | Path,
    artifacts_root: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8501,
    reload: bool = False,
    title: str = "Retail Forecast Studio",
) -> None:
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Web dependencies are missing. Install them with: pip install -e '.[web]'"
        ) from exc

    app = create_app(data_root, artifacts_root, title=title)
    uvicorn.run(app, host=host, port=port, reload=reload)
