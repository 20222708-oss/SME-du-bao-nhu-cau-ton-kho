from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .models import forecast_single_group


def _safe_mape(actual: pd.Series, forecast: pd.Series) -> pd.Series:
    actual = actual.astype(float)
    forecast = forecast.astype(float)
    denominator = actual.abs()
    return np.where(denominator > 1e-6, (actual - forecast).abs() / denominator * 100, np.nan)


def _safe_smape(actual: pd.Series, forecast: pd.Series) -> pd.Series:
    actual = actual.astype(float)
    forecast = forecast.astype(float)
    denominator = actual.abs() + forecast.abs()
    return np.where(denominator > 1e-6, 2 * (actual - forecast).abs() / denominator * 100, np.nan)


def _first_present(frame: pd.DataFrame, columns: list[str]) -> dict[str, object]:
    if frame.empty:
        return {}
    row = frame.iloc[0]
    return {column: row[column] for column in columns if column in frame.columns}


def build_model_evaluation_tables(
    history: pd.DataFrame,
    horizon: int = 30,
    max_groups: int | None = None,
    enable_prophet: bool = True,
    enable_lstm: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest forecast models and return detailed and summary metrics.

    The last `horizon` days of each store/product series are held out as
    validation data. Models are trained on the previous observations, forecast
    the holdout dates, then are compared with actual sales. This produces data
    that is easy to explain in Power BI: actual vs forecast, error by model,
    error by product group, error by month, and ranking by MAE/RMSE/MAPE/BIAS.
    """

    history = history.copy().sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    rows: list[pd.DataFrame] = []
    groups = []

    for (store_id, item_id), group in history.groupby(["store_id", "item_id"], sort=False):
        groups.append((store_id, item_id, group.copy()))

    if max_groups is not None:
        groups = sorted(groups, key=lambda item: float(item[2]["target"].sum()), reverse=True)[:max_groups]

    meta_cols = [
        "product_name",
        "category",
        "sub_category",
        "brand",
        "store_name",
        "city",
        "district",
        "store_type",
    ]

    for store_id, item_id, group in groups:
        group = group.sort_values("date").reset_index(drop=True)
        eval_len = min(horizon, max(7, len(group) // 5))
        if len(group) <= eval_len + 14:
            continue

        train_part = group.iloc[:-eval_len].copy()
        actual_part = group.iloc[-eval_len:][["date", "target"]].copy()
        try:
            forecasts = forecast_single_group(
                train_part,
                horizon=eval_len,
                enable_prophet=enable_prophet,
                enable_lstm=enable_lstm,
            )
        except Exception:
            continue

        metadata = _first_present(group, meta_cols)
        for model_name, forecast in forecasts.items():
            if forecast is None or forecast.empty:
                continue
            compare = forecast[["date", "forecast"]].copy()
            compare["date"] = pd.to_datetime(compare["date"])
            compare = compare.merge(actual_part, on="date", how="inner")
            if compare.empty:
                continue

            compare["store_id"] = store_id
            compare["item_id"] = item_id
            compare["model_name"] = model_name
            compare = compare.rename(columns={"target": "actual"})
            compare["error"] = compare["forecast"].astype(float) - compare["actual"].astype(float)
            compare["abs_error"] = compare["error"].abs()
            compare["squared_error"] = compare["error"] ** 2
            compare["ape"] = _safe_mape(compare["actual"], compare["forecast"])
            compare["smape"] = _safe_smape(compare["actual"], compare["forecast"])
            compare["bias_pct"] = np.where(
                compare["actual"].abs() > 1e-6,
                compare["error"] / compare["actual"].abs() * 100,
                np.nan,
            )
            for key, value in metadata.items():
                compare[key] = value
            rows.append(compare)

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    detail = pd.concat(rows, ignore_index=True)
    detail["year"] = pd.to_datetime(detail["date"]).dt.year
    detail["month"] = pd.to_datetime(detail["date"]).dt.month

    summary = (
        detail.groupby("model_name", dropna=False)
        .agg(
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda s: float(np.sqrt(s.mean()))),
            mape=("ape", "mean"),
            smape=("smape", "mean"),
            bias=("bias_pct", "mean"),
            rows=("date", "size"),
            groups=("item_id", "nunique"),
            mean_actual=("actual", "mean"),
            mean_forecast=("forecast", "mean"),
        )
        .reset_index()
    )
    summary = summary.sort_values(["smape", "mae"], na_position="last").reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return detail, summary


def build_model_performance_orientation(summary: pd.DataFrame) -> pd.DataFrame:
    """Create a business-friendly comparison table for forecast models.

    The regular summary table is numeric. This table adds interpretation fields
    so the Power BI model page and graduation slides can explain when each model
    should be used, not only which model has the lowest error.
    """

    profiles = {
        "baseline": {
            "ten_hien_thi": "Baseline",
            "vai_tro": "Moc so sanh ban dau",
            "toc_do_huan_luyen": "Rat nhanh",
            "kha_nang_giai_thich": "Cao",
            "phu_hop_mua_vu": "Thap",
            "dinh_huong_su_dung": "Dung lam moc nen de biet cac mo hinh khac co cai thien hay khong.",
            "han_che": "Kho bat duoc mua vu, ngay le va bien dong phuc tap.",
        },
        "regression": {
            "ten_hien_thi": "Regression",
            "vai_tro": "Mo hinh hoc may nen",
            "toc_do_huan_luyen": "Nhanh",
            "kha_nang_giai_thich": "Cao",
            "phu_hop_mua_vu": "Trung binh",
            "dinh_huong_su_dung": "Phu hop khi can ket qua nhanh, de giai thich va lam baseline.",
            "han_che": "Can dac trung thoi gian tot, kho hoc quan he chuoi dai han.",
        },
        "prophet": {
            "ten_hien_thi": "Prophet",
            "vai_tro": "Du bao chuoi thoi gian",
            "toc_do_huan_luyen": "Trung binh",
            "kha_nang_giai_thich": "Kha cao",
            "phu_hop_mua_vu": "Cao",
            "dinh_huong_su_dung": "Phu hop du lieu co xu huong, mua vu, ngay le va Tet.",
            "han_che": "Can chuoi thoi gian tuong doi on dinh, co the kem linh hoat voi bien dong bat thuong.",
        },
        "lstm": {
            "ten_hien_thi": "LSTM",
            "vai_tro": "Hoc sau cho chuoi thoi gian",
            "toc_do_huan_luyen": "Cham",
            "kha_nang_giai_thich": "Thap",
            "phu_hop_mua_vu": "Kha cao",
            "dinh_huong_su_dung": "Phu hop khi du lieu dai, nhieu mau bien dong va can hoc quan he phuc tap.",
            "han_che": "Ton thoi gian train, kho giai thich va can nhieu du lieu.",
        },
        "ensemble": {
            "ten_hien_thi": "Ensemble",
            "vai_tro": "Ket hop nhieu mo hinh",
            "toc_do_huan_luyen": "Phu thuoc mo hinh thanh phan",
            "kha_nang_giai_thich": "Trung binh",
            "phu_hop_mua_vu": "Cao",
            "dinh_huong_su_dung": "Phu hop khi can ket qua on dinh va giam rui ro phu thuoc mot mo hinh.",
            "han_che": "Kho giai thich hon mo hinh don le.",
        },
    }

    if summary.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "ten_hien_thi",
                "rank",
                "mae",
                "rmse",
                "mape",
                "smape",
                "bias",
                "muc_hieu_nang",
                "xu_huong_du_bao",
                "vai_tro",
                "toc_do_huan_luyen",
                "kha_nang_giai_thich",
                "phu_hop_mua_vu",
                "dinh_huong_su_dung",
                "han_che",
                "ket_luan",
            ]
        )

    rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        model_name = str(row.get("model_name", "")).lower()
        profile = profiles.get(
            model_name,
            {
                "ten_hien_thi": str(row.get("model_name", "")),
                "vai_tro": "Mo hinh du bao",
                "toc_do_huan_luyen": "Chua danh gia",
                "kha_nang_giai_thich": "Chua danh gia",
                "phu_hop_mua_vu": "Chua danh gia",
                "dinh_huong_su_dung": "Can xem them chi so thuc nghiem truoc khi lua chon.",
                "han_che": "Can danh gia them tren du lieu thuc te.",
            },
        )
        smape = float(row["smape"]) if pd.notna(row.get("smape")) else np.nan
        bias = float(row["bias"]) if pd.notna(row.get("bias")) else np.nan

        if pd.isna(smape):
            performance_level = "Chua du du lieu"
        elif smape <= 15:
            performance_level = "Tot"
        elif smape <= 30:
            performance_level = "Kha"
        elif smape <= 50:
            performance_level = "Trung binh"
        else:
            performance_level = "Can cai thien"

        if pd.isna(bias) or abs(bias) < 1:
            bias_direction = "Can bang"
        elif bias > 0:
            bias_direction = "Du bao cao hon thuc te"
        else:
            bias_direction = "Du bao thap hon thuc te"

        rank = int(row["rank"]) if pd.notna(row.get("rank")) else None
        if rank == 1:
            conclusion = "Nen uu tien theo ket qua thuc nghiem hien tai."
        elif model_name in {"baseline", "regression"}:
            conclusion = "Nen dung de doi chieu va giai thich muc cai thien."
        else:
            conclusion = "Co the dung khi phu hop boi canh du lieu va yeu cau nghiep vu."

        rows.append(
            {
                "model_name": row.get("model_name"),
                "ten_hien_thi": profile["ten_hien_thi"],
                "rank": row.get("rank"),
                "mae": row.get("mae"),
                "rmse": row.get("rmse"),
                "mape": row.get("mape"),
                "smape": row.get("smape"),
                "bias": row.get("bias"),
                "muc_hieu_nang": performance_level,
                "xu_huong_du_bao": bias_direction,
                "vai_tro": profile["vai_tro"],
                "toc_do_huan_luyen": profile["toc_do_huan_luyen"],
                "kha_nang_giai_thich": profile["kha_nang_giai_thich"],
                "phu_hop_mua_vu": profile["phu_hop_mua_vu"],
                "dinh_huong_su_dung": profile["dinh_huong_su_dung"],
                "han_che": profile["han_che"],
                "ket_luan": conclusion,
            }
        )

    return pd.DataFrame(rows).sort_values(["rank", "model_name"], na_position="last").reset_index(drop=True)


def write_evaluation_outputs(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    detail_path = output / "model_evaluation_detail.csv"
    summary_path = output / "model_evaluation_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    paths["model_evaluation_detail"] = detail_path
    paths["model_evaluation_summary"] = summary_path

    orientation = build_model_performance_orientation(summary)
    orientation_path = output / "model_performance_orientation.csv"
    orientation.to_csv(orientation_path, index=False)
    paths["model_performance_orientation"] = orientation_path

    if not detail.empty:
        if "category" in detail.columns:
            by_category = (
                detail.groupby(["category", "model_name"], dropna=False)
                .agg(
                    mae=("abs_error", "mean"),
                    mape=("ape", "mean"),
                    smape=("smape", "mean"),
                    rows=("date", "size"),
                )
                .reset_index()
                .sort_values(["category", "smape"])
            )
        else:
            by_category = pd.DataFrame(columns=["category", "model_name", "mae", "mape", "smape", "rows"])

        by_month = (
            detail.groupby(["year", "month", "model_name"], dropna=False)
            .agg(mae=("abs_error", "mean"), mape=("ape", "mean"), smape=("smape", "mean"), rows=("date", "size"))
            .reset_index()
            .sort_values(["year", "month", "model_name"])
        )
        category_path = output / "model_error_by_category.csv"
        month_path = output / "model_error_by_month.csv"
        by_category.to_csv(category_path, index=False)
        by_month.to_csv(month_path, index=False)
        paths["model_error_by_category"] = category_path
        paths["model_error_by_month"] = month_path

    return paths


def generate_evaluation_charts(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Create PNG charts for reports. Skips gracefully when matplotlib is absent."""

    if detail.empty or summary.empty:
        return {}

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    output = Path(output_dir) / "charts"
    output.mkdir(parents=True, exist_ok=True)
    chart_paths: dict[str, Path] = {}

    palette = {
        "baseline": "#6B7280",
        "regression": "#6B7280",
        "prophet": "#2563EB",
        "lstm": "#F59E0B",
        "ensemble": "#007A55",
    }

    def _save(fig, name: str) -> None:
        path = output / name
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        chart_paths[name.removesuffix(".png")] = path

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ordered = summary.sort_values("smape")
    colors = [palette.get(str(name).lower(), "#007A55") for name in ordered["model_name"]]
    ax.bar(ordered["model_name"], ordered["smape"], color=colors)
    ax.set_title("So sánh sai số SMAPE theo mô hình")
    ax.set_ylabel("SMAPE (%)")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "model_smape_ranking.png")

    metric_cols = [column for column in ["mae", "rmse"] if column in summary.columns]
    if metric_cols:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        x = np.arange(len(summary["model_name"]))
        width = 0.35
        for idx, metric in enumerate(metric_cols):
            offset = (idx - (len(metric_cols) - 1) / 2) * width
            ax.bar(x + offset, summary[metric], width=width, label=metric.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(summary["model_name"])
        ax.set_title("So sánh MAE và RMSE theo mô hình")
        ax.set_ylabel("Sai số")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        _save(fig, "model_mae_rmse_comparison.png")

    if "bias" in summary.columns:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        bias_values = summary["bias"].fillna(0)
        colors = np.where(bias_values >= 0, "#007A55", "#D64545")
        ax.bar(summary["model_name"], bias_values, color=colors)
        ax.axhline(0, color="#374151", linewidth=1)
        ax.set_title("Độ lệch dự báo BIAS theo mô hình")
        ax.set_ylabel("BIAS (%)")
        ax.grid(axis="y", alpha=0.25)
        _save(fig, "model_bias_comparison.png")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(detail["ape"].dropna().clip(upper=100), bins=12, color="#007A55", edgecolor="white")
    ax.set_title("Phân bố sai số dự báo APE")
    ax.set_xlabel("APE (%)")
    ax.set_ylabel("Số bản ghi")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "forecast_error_distribution.png")

    best_model = str(summary.iloc[0]["model_name"])
    sample_key = (
        detail[detail["model_name"] == best_model]
        .groupby(["store_id", "item_id"], dropna=False)["actual"]
        .sum()
        .sort_values(ascending=False)
        .head(1)
    )
    if not sample_key.empty:
        store_id, item_id = sample_key.index[0]
        sample = detail[
            (detail["store_id"] == store_id)
            & (detail["item_id"] == item_id)
            & (detail["model_name"] == best_model)
        ].sort_values("date")
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(sample["date"], sample["actual"], label="Thực tế", color="#0F4C5C", linewidth=2.4)
        ax.plot(sample["date"], sample["forecast"], label=f"Dự báo {best_model}", color="#F59E0B", linewidth=2.2)
        title_product = sample["product_name"].dropna().iloc[0] if "product_name" in sample and sample["product_name"].notna().any() else item_id
        ax.set_title(f"Thực tế và dự báo - {title_product}")
        ax.set_ylabel("Số lượng")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        _save(fig, "actual_vs_forecast_sample.png")

        fig, ax = plt.subplots(figsize=(6.8, 6.2))
        ax.scatter(sample["actual"], sample["forecast"], color="#007A55", alpha=0.75)
        limit = max(float(sample["actual"].max()), float(sample["forecast"].max()))
        ax.plot([0, limit], [0, limit], color="#D64545", linestyle="--", linewidth=1.5, label="Dự báo hoàn hảo")
        ax.set_title(f"Thực tế so với dự báo - {best_model}")
        ax.set_xlabel("Thực tế")
        ax.set_ylabel("Dự báo")
        ax.legend()
        ax.grid(alpha=0.25)
        _save(fig, "actual_vs_forecast_scatter.png")

    daily_error = (
        detail[detail["model_name"] == best_model]
        .groupby("date", dropna=False)["abs_error"]
        .mean()
        .reset_index()
        .sort_values("date")
    )
    if not daily_error.empty:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(daily_error["date"], daily_error["abs_error"], color="#D64545", linewidth=2)
        ax.set_title(f"Xu hướng sai số tuyệt đối theo ngày - {best_model}")
        ax.set_ylabel("Sai số tuyệt đối trung bình")
        ax.grid(axis="y", alpha=0.25)
        _save(fig, "daily_absolute_error_trend.png")

    if "category" in detail.columns:
        category = (
            detail[detail["model_name"] == best_model]
            .groupby("category", dropna=False)["smape"]
            .mean()
            .sort_values()
        )
        if not category.empty:
            fig, ax = plt.subplots(figsize=(9, 4.8))
            ax.barh(category.index.astype(str), category.values, color="#007A55")
            ax.set_title(f"SMAPE theo nhóm sản phẩm - {best_model}")
            ax.set_xlabel("SMAPE (%)")
            ax.grid(axis="x", alpha=0.25)
            _save(fig, "model_error_by_category.png")

    item_label = "product_name" if "product_name" in detail.columns else "item_id"
    top_items = (
        detail[detail["model_name"] == best_model]
        .groupby(item_label, dropna=False)["smape"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    if not top_items.empty:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.barh(top_items.index.astype(str), top_items.values, color="#D64545")
        ax.invert_yaxis()
        ax.set_title(f"Top SKU có sai số SMAPE cao - {best_model}")
        ax.set_xlabel("SMAPE (%)")
        ax.grid(axis="x", alpha=0.25)
        _save(fig, "top_sku_error.png")

    month = (
        detail[detail["model_name"] == best_model]
        .groupby(["year", "month"], dropna=False)["smape"]
        .mean()
        .reset_index()
    )
    if not month.empty:
        pivot = month.pivot(index="year", columns="month", values="smape").sort_index()
        fig, ax = plt.subplots(figsize=(10, 4.8))
        im = ax.imshow(pivot.fillna(0), cmap="RdYlGn_r", aspect="auto")
        ax.set_title(f"Heatmap sai số SMAPE theo tháng - {best_model}")
        ax.set_xlabel("Tháng")
        ax.set_ylabel("Năm")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(i) for i in pivot.index])
        fig.colorbar(im, ax=ax, label="SMAPE (%)")
        _save(fig, "model_error_heatmap_by_month.png")

    return chart_paths
