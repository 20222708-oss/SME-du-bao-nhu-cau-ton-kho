from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .features import prepare_supervised_frame
from .models import load_bundle
from .webapp import DashboardRepository, _compact_number, _filter_frame, _first_existing


DEFAULT_DATA_ROOT = Path(os.environ.get("RETAIL_FORECAST_DATA_ROOT", r"D:/retail_artifacts"))
DEFAULT_TITLE = "Retail Forecast Studio"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--artifacts-root", default=None)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    args, _ = parser.parse_known_args()
    return args


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f6f9ff 0%, #eef3fb 100%);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0e2a57 0%, #123d78 100%);
            color: #f8fbff;
        }
        [data-testid="stSidebar"] * {
            color: #f8fbff !important;
        }
        .hero {
            background: linear-gradient(135deg, #0f2d5c 0%, #123d78 100%);
            color: white;
            padding: 18px 22px;
            border-radius: 22px;
            box-shadow: 0 16px 36px rgba(15, 45, 92, 0.16);
            margin-bottom: 18px;
        }
        .hero h1 {
            margin: 0;
            font-size: 2rem;
            font-weight: 800;
        }
        .hero p {
            margin: 6px 0 0 0;
            opacity: 0.9;
        }
        .section-title {
            font-size: 1.05rem;
            font-weight: 800;
            color: #123d78;
            margin: 0.2rem 0 0.6rem 0;
        }
        .kpi-card {
            background: #ffffff;
            border-radius: 18px;
            padding: 16px 18px;
            border: 1px solid #dfe8f6;
            box-shadow: 0 10px 28px rgba(15, 45, 92, 0.08);
            min-height: 110px;
        }
        .kpi-label {
            font-size: 0.73rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #5f7aa8;
            font-weight: 800;
        }
        .kpi-value {
            font-size: 1.9rem;
            font-weight: 900;
            color: #0f2d5c;
            line-height: 1.1;
            margin-top: 6px;
        }
        .kpi-sub {
            color: #6b7a90;
            font-size: 0.9rem;
            margin-top: 6px;
        }
        .panel {
            background: rgba(255,255,255,0.95);
            border: 1px solid #dfe8f6;
            border-radius: 20px;
            padding: 18px;
            box-shadow: 0 10px 28px rgba(15, 45, 92, 0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _card(label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """


def _render_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(_card(label, value, sub), unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def _repo(data_root: str, artifacts_root: str | None) -> DashboardRepository:
    return DashboardRepository(Path(data_root), Path(artifacts_root) if artifacts_root else None)


def _load_json_metrics(roots: list[Path]) -> dict[str, pd.DataFrame]:
    metrics_path = _first_existing(roots, ["evaluation_metrics.csv"])
    if metrics_path is not None:
        frame = pd.read_csv(metrics_path, low_memory=False)
        for col in ["mae", "rmse", "mape", "bias", "mean_forecast", "coverage_groups", "rows", "groups"]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return {"evaluation": frame}

    json_path = _first_existing(roots, ["metrics.json"])
    if json_path is None:
        return {"evaluation": pd.DataFrame()}

    try:
        data = pd.read_json(json_path, typ="series").to_dict()
    except Exception:
        import json

        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

    rows: list[dict[str, object]] = []
    regression = data.get("regression", {}) if isinstance(data, dict) else {}
    if regression:
        rows.append(
            {
                "section": "regression",
                "model_name": "regression",
                "mae": regression.get("mae"),
                "rmse": regression.get("rmse"),
                "mape": regression.get("mape"),
                "bias": regression.get("bias"),
                "mean_forecast": None,
                "coverage_groups": None,
            }
        )

    forecast_models = data.get("forecast_models", {}) if isinstance(data, dict) else {}
    coverage = data.get("coverage", {}) if isinstance(data, dict) else {}
    for name, mean_forecast in forecast_models.items():
        rows.append(
            {
                "section": "forecast_models",
                "model_name": name,
                "mae": None,
                "rmse": None,
                "mape": None,
                "bias": None,
                "mean_forecast": mean_forecast,
                "coverage_groups": coverage.get(name),
            }
        )

    return {"evaluation": pd.DataFrame(rows)}


def _compute_regression_metrics(history: pd.DataFrame, artifacts_root: Path | None) -> dict[str, float]:
    if artifacts_root is None:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "bias": 0.0}

    bundle_path = _first_existing([artifacts_root], ["regression_bundle.joblib"])
    if bundle_path is None:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "bias": 0.0}

    try:
        bundle = load_bundle(str(bundle_path))
        frame = prepare_supervised_frame(history)
        if frame.empty or "target" not in frame.columns:
            return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "bias": 0.0}
        pred = bundle.predict(frame)
        actual = frame["target"].astype(float).to_numpy()
        pred = np.asarray(pred, dtype=float)
        denom = np.clip(np.abs(actual), 1e-6, None)
        return {
            "mae": float(np.mean(np.abs(actual - pred))),
            "rmse": float(np.sqrt(np.mean((actual - pred) ** 2))),
            "mape": float(np.mean(np.abs((actual - pred) / denom)) * 100),
            "bias": float(np.mean((pred - actual) / denom) * 100),
        }
    except Exception:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "bias": 0.0}


def _available_models(forecast: pd.DataFrame, metrics: pd.DataFrame) -> list[str]:
    models: list[str] = []
    if not forecast.empty and "model_name" in forecast.columns:
        models.extend(sorted(forecast["model_name"].dropna().astype(str).unique().tolist()))
    if not metrics.empty and "model_name" in metrics.columns:
        models.extend([str(v) for v in metrics["model_name"].dropna().tolist()])
    models = sorted(set(models))
    if "ensemble" in models:
        models.remove("ensemble")
        models.insert(0, "ensemble")
    return models or ["ensemble", "prophet", "lstm", "baseline"]


def _build_actual_forecast_figure(history: pd.DataFrame, forecast: pd.DataFrame, selected_models: list[str]) -> go.Figure:
    fig = go.Figure()

    if not history.empty and {"date", "target"} <= set(history.columns):
        actual = history.groupby("date", as_index=False)["target"].sum().sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=actual["date"],
                y=actual["target"],
                mode="lines+markers",
                name="Thực tế",
                line=dict(color="#0f7a8a", width=3),
            )
        )

    if not forecast.empty and {"date", "forecast"} <= set(forecast.columns):
        if "model_name" in forecast.columns and selected_models != ["Tất cả"]:
            forecast = forecast[forecast["model_name"].astype(str).isin(selected_models)].copy()
        if "model_name" in forecast.columns and (selected_models == ["Tất cả"] or len(selected_models) > 1):
            for model_name, group in forecast.groupby("model_name", dropna=False):
                grp = group.groupby("date", as_index=False)["forecast"].sum().sort_values("date")
                fig.add_trace(
                    go.Scatter(
                        x=grp["date"],
                        y=grp["forecast"],
                        mode="lines",
                        name=f"Dự báo · {model_name}",
                        line=dict(width=2),
                    )
                )
        else:
            grp = forecast.groupby("date", as_index=False)["forecast"].sum().sort_values("date")
            fig.add_trace(
                go.Scatter(
                    x=grp["date"],
                    y=grp["forecast"],
                    mode="lines",
                    name="Dự báo",
                    line=dict(color="#2f6fed", width=3, dash="dash"),
                )
            )

        if {"yhat_lower", "yhat_upper"} <= set(forecast.columns) and "model_name" in forecast.columns and selected_models != ["Tất cả"]:
            selected = forecast[forecast["model_name"].astype(str).isin(selected_models)]
            band = selected.groupby("date", as_index=False)[["yhat_lower", "yhat_upper"]].mean().sort_values("date")
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([band["date"], band["date"][::-1]]),
                    y=pd.concat([band["yhat_upper"], band["yhat_lower"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(47,111,237,0.18)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Khoảng tin cậy",
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

    fig.update_layout(
        margin=dict(l=10, r=10, t=25, b=10),
        height=430,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="Ngày",
        yaxis_title="Đơn vị",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(15,45,92,0.08)")
    return fig


def _build_top_products(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty or not {"product_name", "category", "target"} <= set(history.columns):
        return pd.DataFrame()
    out = (
        history.groupby(["product_name", "category"], as_index=False)
        .agg(
            **{
                "Doanh thu": ("revenue", "sum") if "revenue" in history.columns else ("target", "sum"),
                "Số lượng": ("target", "sum"),
            }
        )
        .sort_values("Doanh thu", ascending=False)
        .head(10)
    )
    return out.reset_index(drop=True)


def _build_status_table(history: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame()
    latest_stock = pd.DataFrame()
    if not history.empty and {"store_id", "item_id", "date", "stock_end"} <= set(history.columns):
        latest_stock = (
            history.sort_values("date")
            .groupby(["store_id", "item_id"], as_index=False)
            .tail(1)[["store_id", "item_id", "stock_end"]]
            .drop_duplicates(["store_id", "item_id"])
        )
    merged = inventory.copy()
    if not latest_stock.empty:
        merged = merged.merge(latest_stock, on=["store_id", "item_id"], how="left")

    if "stock_end" not in merged.columns:
        merged["stock_end"] = 0

    stock_end = pd.to_numeric(merged["stock_end"], errors="coerce").fillna(0)
    reorder_point = pd.to_numeric(merged.get("reorder_point", pd.Series(0, index=merged.index)), errors="coerce").fillna(0)
    safety_stock = pd.to_numeric(merged.get("safety_stock", pd.Series(0, index=merged.index)), errors="coerce").fillna(0)

    merged["Trạng thái tồn kho"] = np.select(
        [stock_end >= reorder_point, stock_end >= safety_stock],
        ["Đủ hàng", "Cần nhập"],
        default="Thiếu hàng",
    )
    return merged


def _render_metric_row(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        with col:
            _render_card(label, value, sub)


def main() -> None:
    args = _parse_args()
    st.set_page_config(page_title=args.title, page_icon="📈", layout="wide")
    _inject_css()

    st.markdown(
        f"""
        <div class="hero">
            <h1>{args.title}</h1>
            <p>Dashboard dự báo nhu cầu, tối ưu tồn kho và đánh giá mô hình từ dữ liệu bán lẻ Việt Nam.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Cấu hình dữ liệu")
        data_root_text = st.text_input("Thư mục dữ liệu", value=str(args.data_root))
        artifacts_root_text = st.text_input("Thư mục artifacts (tuỳ chọn)", value=args.artifacts_root or "")
        st.caption("Mặc định dùng `D:/retail_artifacts`. Nếu có thư mục khác, dán đường dẫn vào đây.")

    data_root = Path(data_root_text).expanduser()
    artifacts_root = Path(artifacts_root_text).expanduser() if artifacts_root_text.strip() else None

    if not data_root.exists():
        st.error(f"Không tìm thấy thư mục dữ liệu: {data_root}")
        st.stop()

    repo = _repo(str(data_root), str(artifacts_root) if artifacts_root else None)
    history = repo.history.copy()
    forecast = repo.forecast.copy()
    inventory = repo.inventory.copy()
    products = repo.products.copy()
    stores = repo.stores.copy()
    metrics = repo.metrics.copy()
    eval_frame = _load_json_metrics([p for p in [artifacts_root, data_root] if p is not None])["evaluation"]

    if history.empty:
        st.error("Không tải được bảng lịch sử bán hàng từ thư mục đã chọn.")
        st.stop()

    model_options = _available_models(forecast, metrics)
    if "all" not in model_options:
        model_options = ["Tất cả"] + model_options

    sidebar_col1, sidebar_col2 = st.sidebar.columns(2)
    with sidebar_col1:
        selected_store = st.selectbox(
            "Kho / Cửa hàng",
            options=["all"] + sorted(history["store_id"].dropna().astype(str).unique().tolist()),
            format_func=lambda x: "Tất cả" if x == "all" else x,
        )
    with sidebar_col2:
        selected_category = st.selectbox(
            "Nhóm SP",
            options=["all"] + sorted(history["category"].dropna().astype(str).unique().tolist()) if "category" in history.columns else ["all"],
            format_func=lambda x: "Tất cả" if x == "all" else x,
        )
    selected_item = st.sidebar.selectbox(
        "Sản phẩm",
        options=["all"] + sorted(history["item_id"].dropna().astype(str).unique().tolist()),
        format_func=lambda x: "Tất cả" if x == "all" else x,
    )
    selected_model = st.sidebar.selectbox(
        "Mô hình",
        options=model_options,
        index=model_options.index("ensemble") if "ensemble" in model_options else 0,
    )
    horizon = st.sidebar.slider("Số ngày dự báo", min_value=7, max_value=90, value=30, step=1)
    history_window = st.sidebar.slider("Cửa sổ lịch sử", min_value=30, max_value=365, value=120, step=10)

    history_f = _filter_frame(
        history,
        store_id=None if selected_store == "all" else selected_store,
        item_id=None if selected_item == "all" else selected_item,
        category=None if selected_category == "all" else selected_category,
    )
    forecast_f = _filter_frame(
        forecast,
        store_id=None if selected_store == "all" else selected_store,
        item_id=None if selected_item == "all" else selected_item,
        category=None if selected_category == "all" else selected_category,
        model_name=None if selected_model == "Tất cả" else selected_model,
    )
    inventory_f = _filter_frame(
        inventory,
        store_id=None if selected_store == "all" else selected_store,
        item_id=None if selected_item == "all" else selected_item,
        category=None if selected_category == "all" else selected_category,
    )

    if selected_model == "Tất cả" and "model_name" in forecast_f.columns:
        selected_models = ["Tất cả"]
    else:
        selected_models = [selected_model]

    latest_stock = _build_status_table(history_f, inventory_f)
    regression_metrics = _compute_regression_metrics(history, artifacts_root)

    total_units = float(history_f["target"].sum()) if "target" in history_f.columns else 0.0
    total_revenue = float(history_f["revenue"].sum()) if "revenue" in history_f.columns else 0.0
    total_forecast = float(forecast_f["forecast"].sum()) if "forecast" in forecast_f.columns else 0.0
    product_count = int(history_f["item_id"].nunique()) if "item_id" in history_f.columns else 0
    store_count = int(history_f["store_id"].nunique()) if "store_id" in history_f.columns else 0
    stockout_rate = float(history_f["stockout_flag"].mean() * 100) if "stockout_flag" in history_f.columns and not history_f.empty else 0.0
    safety_stock_avg = float(inventory_f["safety_stock"].mean()) if "safety_stock" in inventory_f.columns and not inventory_f.empty else 0.0
    reorder_point_avg = float(inventory_f["reorder_point"].mean()) if "reorder_point" in inventory_f.columns and not inventory_f.empty else 0.0
    eoq_avg = float(inventory_f["eoq"].mean()) if "eoq" in inventory_f.columns and not inventory_f.empty else 0.0

    actual_window = (
        history_f.sort_values("date")
        .tail(history_window)
        if "date" in history_f.columns and not history_f.empty
        else pd.DataFrame()
    )
    recent_units = float(actual_window["target"].sum()) if "target" in actual_window.columns and not actual_window.empty else 0.0
    forecast_ratio = (total_forecast / recent_units * 100) if recent_units else 0.0

    st.markdown("### Tổng quan nhanh")
    _render_metric_row(
        [
            ("Tổng thực tế", _compact_number(total_units), f"{_compact_number(total_revenue)} doanh thu"),
            ("Tổng dự báo", _compact_number(total_forecast), f"{horizon} ngày tới"),
            ("Tồn kho an toàn TB", _compact_number(safety_stock_avg), "mức dự phòng"),
            ("Số SKU cần đặt", _compact_number(len(inventory_f)), "SKU trong danh sách"),
        ]
    )

    tab_overview, tab_forecast, tab_inventory, tab_model, tab_data = st.tabs(
        ["Tổng quan", "Dự báo theo sản phẩm", "Tồn kho & đặt hàng", "Mô hình", "Dữ liệu"]
    )

    with tab_overview:
        left, right = st.columns([2.1, 1.1], gap="large")
        with left:
            st.markdown('<div class="section-title">Thực tế vs Dự báo</div>', unsafe_allow_html=True)
            st.plotly_chart(
                _build_actual_forecast_figure(history_f, forecast_f, selected_models),
                use_container_width=True,
                theme=None,
            )
        with right:
            st.markdown('<div class="section-title">Chỉ số tổng quan</div>', unsafe_allow_html=True)
            k1, k2 = st.columns(2)
            with k1:
                st.metric("Sản phẩm", _compact_number(product_count))
                st.metric("Tỷ lệ stockout", f"{stockout_rate:.2f}%")
            with k2:
                st.metric("Cửa hàng", _compact_number(store_count))
                st.metric("Forecast / 30 ngày", f"{forecast_ratio:.1f}%")

            if "category" in history_f.columns:
                cat = (
                    history_f.groupby("category", as_index=False)["target"].sum().sort_values("target", ascending=False)
                )
                if not cat.empty:
                    fig = px.bar(cat.head(6), x="target", y="category", orientation="h", title="Doanh số theo danh mục")
                    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Top sản phẩm bán chạy</div>', unsafe_allow_html=True)
        top_products = _build_top_products(history_f)
        if top_products.empty:
            st.info("Chưa có đủ dữ liệu để dựng bảng Top sản phẩm.")
        else:
            st.dataframe(top_products, use_container_width=True, hide_index=True)

    with tab_forecast:
        left, right = st.columns([2.2, 1], gap="large")
        with left:
            st.markdown('<div class="section-title">Dự báo nhu cầu</div>', unsafe_allow_html=True)
            st.plotly_chart(
                _build_actual_forecast_figure(history_f, forecast_f, selected_models),
                use_container_width=True,
                theme=None,
            )
            st.markdown('<div class="section-title">Dự báo theo ngày</div>', unsafe_allow_html=True)
            table = forecast_f.copy()
            if not table.empty and "date" in table.columns:
                table = table.sort_values("date").tail(30)
                show_cols = [c for c in ["date", "forecast", "model_name", "yhat_lower", "yhat_upper"] if c in table.columns]
                st.dataframe(table[show_cols], use_container_width=True, hide_index=True)
            else:
                st.info("Chưa có dữ liệu forecast để hiển thị.")
        with right:
            st.markdown('<div class="section-title">Đánh giá mô hình</div>', unsafe_allow_html=True)
            if not np.isclose(regression_metrics["mae"], 0.0) or not np.isclose(regression_metrics["rmse"], 0.0):
                st.metric("MAE", _compact_number(regression_metrics["mae"]))
                st.metric("RMSE", _compact_number(regression_metrics["rmse"]))
                st.metric("MAPE", f'{regression_metrics["mape"]:.2f}%')
                st.metric("BIAS", f'{regression_metrics["bias"]:.2f}%')
            else:
                st.info("Chưa tính được metric từ model bundle, sẽ hiển thị từ file metrics nếu có.")

            if not eval_frame.empty and {"model_name", "section"} <= set(eval_frame.columns):
                eval_models = eval_frame[eval_frame["section"].astype(str) == "forecast_models"].copy()
                if not eval_models.empty and "mean_forecast" in eval_models.columns:
                    fig = px.bar(
                        eval_models,
                        x="model_name",
                        y="mean_forecast",
                        title="Mean forecast theo model",
                        color="model_name",
                    )
                    fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10), template="plotly_white", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

    with tab_inventory:
        top1, top2, top3 = st.columns(3)
        with top1:
            st.markdown(_card("Safety Stock TB", _compact_number(safety_stock_avg), "đơn vị"), unsafe_allow_html=True)
        with top2:
            st.markdown(_card("Reorder Point TB", _compact_number(reorder_point_avg), "đơn vị"), unsafe_allow_html=True)
        with top3:
            st.markdown(_card("EOQ TB", _compact_number(eoq_avg), "đơn vị"), unsafe_allow_html=True)

        left, mid, right = st.columns([1.1, 1.6, 1.0], gap="large")
        with left:
            st.markdown('<div class="section-title">Tỷ lệ nhóm tồn kho</div>', unsafe_allow_html=True)
            if latest_stock.empty:
                st.info("Chưa có dữ liệu tồn kho để phân loại.")
            else:
                donut = latest_stock["Trạng thái tồn kho"].value_counts().reset_index()
                donut.columns = ["Trạng thái", "Số SKU"]
                fig = px.pie(donut, values="Số SKU", names="Trạng thái", hole=0.6, color_discrete_sequence=["#0f7a8a", "#f5a623", "#e45756"])
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), template="plotly_white", legend_title_text="")
                st.plotly_chart(fig, use_container_width=True)
        with mid:
            st.markdown('<div class="section-title">Sản phẩm cần đặt (Top 10)</div>', unsafe_allow_html=True)
            if inventory_f.empty:
                st.info("Chưa có dữ liệu inventory_recommendations.")
            else:
                bar = inventory_f.copy()
                if "product_name" in bar.columns and "reorder_point" in bar.columns:
                    bar = bar.sort_values("reorder_point", ascending=False).head(10)
                    fig = px.bar(
                        bar,
                        x="reorder_point",
                        y="product_name",
                        orientation="h",
                        color="category" if "category" in bar.columns else None,
                        title=None,
                    )
                    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown('<div class="section-title">Tổng quan tồn kho</div>', unsafe_allow_html=True)
            st.metric("SKU thiếu hàng", _compact_number(int((latest_stock["Trạng thái tồn kho"] == "Thiếu hàng").sum()) if not latest_stock.empty else 0))
            st.metric("SKU cần nhập", _compact_number(int((latest_stock["Trạng thái tồn kho"] == "Cần nhập").sum()) if not latest_stock.empty else 0))
            st.metric("SKU đủ hàng", _compact_number(int((latest_stock["Trạng thái tồn kho"] == "Đủ hàng").sum()) if not latest_stock.empty else 0))
            st.metric("Vòng quay tồn kho TB", f"{inventory_f['avg_demand'].mean():.2f}" if "avg_demand" in inventory_f.columns and not inventory_f.empty else "0.00")

        st.markdown('<div class="section-title">Bảng chi tiết tồn kho</div>', unsafe_allow_html=True)
        if latest_stock.empty:
            st.info("Chưa có dữ liệu chi tiết.")
        else:
            show_cols = [
                c
                for c in [
                    "store_name",
                    "product_name",
                    "category",
                    "avg_demand",
                    "stock_end",
                    "safety_stock",
                    "reorder_point",
                    "eoq",
                    "Trạng thái tồn kho",
                ]
                if c in latest_stock.columns
            ]
            st.dataframe(latest_stock[show_cols].sort_values(show_cols[0]), use_container_width=True, hide_index=True)

    with tab_model:
        left, right = st.columns([1.1, 1.3], gap="large")
        with left:
            st.markdown('<div class="section-title">Đánh giá mô hình</div>', unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(_card("MAE", _compact_number(regression_metrics["mae"]), "đơn vị"), unsafe_allow_html=True)
                st.markdown(_card("MAPE", f'{regression_metrics["mape"]:.2f}%', "sai số %"), unsafe_allow_html=True)
            with m2:
                st.markdown(_card("RMSE", _compact_number(regression_metrics["rmse"]), "đơn vị"), unsafe_allow_html=True)
                st.markdown(_card("BIAS", f'{regression_metrics["bias"]:.2f}%', "độ lệch"), unsafe_allow_html=True)

            if not eval_frame.empty and {"section", "model_name"} <= set(eval_frame.columns):
                eval_show = eval_frame.copy()
                st.markdown('<div class="section-title">Bảng metrics</div>', unsafe_allow_html=True)
                st.dataframe(eval_show, use_container_width=True, hide_index=True)
        with right:
            st.markdown('<div class="section-title">Model coverage & forecast</div>', unsafe_allow_html=True)
            if not metrics.empty and {"model_name", "rows", "groups"} <= set(metrics.columns):
                show = metrics.copy()
                for col in ["rows", "groups"]:
                    show[col] = pd.to_numeric(show[col], errors="coerce")
                st.dataframe(show, use_container_width=True, hide_index=True)
            else:
                st.info("Không tìm thấy file model_metrics.csv.")

            if not eval_frame.empty and "mean_forecast" in eval_frame.columns:
                summary = eval_frame[eval_frame.get("section").astype(str) == "forecast_models"].copy()
                if not summary.empty:
                    fig = px.bar(summary, x="model_name", y="mean_forecast", color="model_name", title="Mean forecast theo model")
                    fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), template="plotly_white", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

    with tab_data:
        st.markdown('<div class="section-title">Dữ liệu đã nạp</div>', unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("History rows", _compact_number(len(history)))
            st.metric("Forecast rows", _compact_number(len(forecast)))
        with d2:
            st.metric("Inventory rows", _compact_number(len(inventory)))
            st.metric("Product rows", _compact_number(len(products)))
        with d3:
            st.metric("Store rows", _compact_number(len(stores)))
            st.metric("Metric rows", _compact_number(len(metrics)))

        exp1, exp2, exp3 = st.expander("Xem nhanh bảng lịch sử"), st.expander("Xem nhanh forecast"), st.expander("Xem nhanh tồn kho")
        with exp1:
            st.dataframe(history.head(20), use_container_width=True, hide_index=True)
        with exp2:
            st.dataframe(forecast.head(20), use_container_width=True, hide_index=True)
        with exp3:
            st.dataframe(inventory.head(20), use_container_width=True, hide_index=True)

    st.caption(f"Nguồn dữ liệu chính: {data_root}" + (f" | Artifacts: {artifacts_root}" if artifacts_root else ""))


if __name__ == "__main__":  # pragma: no cover
    main()
