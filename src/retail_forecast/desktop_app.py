from __future__ import annotations

import json
import os
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import pandas as pd

from .webapp import DashboardRepository, _compact_number, _first_existing


DEFAULT_DATA_ROOT = Path(os.environ.get("RETAIL_FORECAST_DATA_ROOT", r"D:/retail_artifacts"))
DEFAULT_TITLE = "Phần mềm dự báo bán lẻ"


def _safe_read_csv(path: Path) -> pd.DataFrame:
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


def _load_evaluation_metrics(roots: list[Path]) -> pd.DataFrame:
    metrics_path = _first_existing(roots, ["evaluation_metrics.csv"])
    if metrics_path is not None:
        frame = _safe_read_csv(metrics_path)
        for col in ["mae", "rmse", "mape", "bias", "mean_forecast", "coverage_groups", "rows", "groups"]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame

    json_path = _first_existing(roots, ["metrics.json"])
    if json_path is None:
        return pd.DataFrame()

    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return pd.DataFrame()

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

    return pd.DataFrame(rows)


def _human_date(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return "-"
    return ts.strftime("%Y-%m-%d")


def _nice_product_label(row: pd.Series) -> str:
    name = str(row.get("product_name") or row.get("item_id") or "").strip()
    return name


def _pretty_table_headers(columns: list[str]) -> list[str]:
    mapping = {
        "section": "Nhóm",
        "model_name": "Mô hình",
        "mae": "Sai số tuyệt đối TB",
        "rmse": "Căn sai số bình phương TB",
        "mape": "Sai số phần trăm TB",
        "bias": "Độ lệch",
        "mean_forecast": "Dự báo trung bình",
        "coverage_groups": "Số nhóm phủ",
        "rows": "Số dòng",
        "groups": "Số nhóm",
        "min_forecast_date": "Ngày dự báo đầu",
        "max_forecast_date": "Ngày dự báo cuối",
        "date": "Ngày",
        "store_id": "Mã cửa hàng",
        "item_id": "Mã sản phẩm",
        "target": "Thực tế",
        "revenue": "Doanh thu",
        "forecast": "Dự báo",
        "store_name": "Cửa hàng",
        "product_name": "Sản phẩm",
        "category": "Nhóm sản phẩm",
        "actual_units": "Thực tế",
        "forecast_units": "Dự báo",
        "avg_demand": "Nhu cầu trung bình",
        "safety_stock": "Tồn kho an toàn",
        "reorder_point": "Điểm đặt hàng lại",
        "eoq": "Lượng đặt hàng tối ưu",
        "source_model": "Nguồn mô hình",
    }
    return [mapping.get(col, col) for col in columns]


def _format_percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return "0%"
    try:
        value = float(value)
    except Exception:
        return "0%"
    if abs(value) >= 10000:
        return "Quá lớn"
    return f"{value:.2f}%"


class ScrollableFrame(ttk.Frame):
    def __init__(self, master: tk.Misc, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, bg=self["background"] if "background" in self.keys() else "#f4f7fb")
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(self.canvas_window, width=event.width),
        )
        self.canvas.bind_all(
            "<MouseWheel>",
            lambda event: self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"),
        )


class SeriesChart(ttk.Frame):
    def __init__(self, master: tk.Misc, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, height=360, bg="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

    def draw(self, actual: pd.DataFrame, forecast: pd.DataFrame, title: str) -> None:
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 800)
        height = max(self.canvas.winfo_height(), 320)
        pad_l, pad_r, pad_t, pad_b = 60, 24, 28, 48

        self.canvas.create_rectangle(8, 8, width - 8, height - 8, outline="#dce6f4", width=1)
        self.canvas.create_text(24, 18, anchor="w", text=title, fill="#163963", font=("Segoe UI", 12, "bold"))

        if actual.empty and forecast.empty:
            self.canvas.create_text(width / 2, height / 2, text="Không có dữ liệu để hiển thị", fill="#6b7a90", font=("Segoe UI", 11))
            return

        actual = actual.copy()
        forecast = forecast.copy()
        if not actual.empty:
            actual["date"] = pd.to_datetime(actual["date"], errors="coerce")
            actual = actual.dropna(subset=["date"]).sort_values("date")
        if not forecast.empty:
            forecast["date"] = pd.to_datetime(forecast["date"], errors="coerce")
            forecast = forecast.dropna(subset=["date"]).sort_values("date")

        frames = []
        if not actual.empty and "value" in actual.columns:
            frames.append(actual["value"].astype(float))
        if not forecast.empty and "value" in forecast.columns:
            frames.append(forecast["value"].astype(float))
        if not frames:
            self.canvas.create_text(width / 2, height / 2, text="Không có cột giá trị phù hợp", fill="#6b7a90", font=("Segoe UI", 11))
            return

        values = pd.concat(frames, ignore_index=True)
        vmin = float(values.min())
        vmax = float(values.max())
        if vmax == vmin:
            vmax += 1.0
            vmin -= 1.0

        combined = pd.concat(
            [actual[["date"]] if not actual.empty else pd.DataFrame(), forecast[["date"]] if not forecast.empty else pd.DataFrame()],
            ignore_index=True,
        )
        dates = []
        if not combined.empty and "date" in combined.columns:
            dates = pd.to_datetime(combined["date"], errors="coerce").dropna().sort_values().drop_duplicates().tolist()
        date_index = {date: idx for idx, date in enumerate(dates)}

        def x_for(idx: int, total: int) -> float:
            if total <= 1:
                return pad_l + 10
            usable = width - pad_l - pad_r
            return pad_l + idx * (usable / (total - 1))

        def y_for(value: float) -> float:
            usable = height - pad_t - pad_b
            return pad_t + (vmax - value) / (vmax - vmin) * usable

        # axes
        self.canvas.create_line(pad_l, pad_t, pad_l, height - pad_b, fill="#d3dceb")
        self.canvas.create_line(pad_l, height - pad_b, width - pad_r, height - pad_b, fill="#d3dceb")

        # y labels
        for tick in range(5):
            value = vmin + (vmax - vmin) * tick / 4
            y = y_for(value)
            self.canvas.create_line(pad_l - 4, y, pad_l, y, fill="#c3cfdf")
            self.canvas.create_text(pad_l - 8, y, anchor="e", text=_compact_number(value), fill="#6b7a90", font=("Segoe UI", 9))

        def draw_line(frame: pd.DataFrame, color: str, label: str) -> None:
            if frame.empty or "value" not in frame.columns:
                return
            points = []
            for _, row in frame.iterrows():
                date = row.get("date")
                if pd.isna(date) or date not in date_index:
                    continue
                points.append((x_for(date_index[date], max(len(dates), 1)), y_for(float(row["value"]))))
            if not points:
                return
            if len(points) > 1:
                self.canvas.create_line(*[coord for point in points for coord in point], fill=color, width=2.5, smooth=True)
            for x, y in points:
                self.canvas.create_oval(x - 2.5, y - 2.5, x + 2.5, y + 2.5, fill=color, outline=color)
            self.canvas.create_rectangle(width - 170, 18 if label == "Thực tế" else 42, width - 158, 30 if label == "Thực tế" else 54, fill=color, outline=color)
            self.canvas.create_text(width - 150, 24 if label == "Thực tế" else 48, anchor="w", text=label, fill="#163963", font=("Segoe UI", 9))

        draw_line(actual, "#0f7a8a", "Thực tế")
        draw_line(forecast, "#2f6ad9", "Dự báo")

        # x labels
        if dates:
            step = max(len(dates) // 6, 1)
            for idx in range(0, len(dates), step):
                x = x_for(idx, len(dates))
                self.canvas.create_text(x, height - pad_b + 18, anchor="n", text=dates[idx].strftime("%d/%m"), fill="#6b7a90", font=("Segoe UI", 8))


class DataTable(ttk.Frame):
    def __init__(self, master: tk.Misc, columns: list[str], *, height: int = 12) -> None:
        super().__init__(master)
        self.columns = columns
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=height)
        yscroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        xscroll = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor="center")

    def set_rows(self, rows: list[dict[str, Any]] | pd.DataFrame, columns: list[str] | None = None) -> None:
        if columns:
            self.columns = columns
            self.tree["columns"] = columns
            self.tree["displaycolumns"] = columns
            for col in columns:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=120, anchor="center")
        for item in self.tree.get_children():
            self.tree.delete(item)

        frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        if frame.empty:
            return
        for _, row in frame.iterrows():
            values = []
            for col in self.columns:
                value = row.get(col, "")
                if pd.isna(value):
                    value = ""
                if isinstance(value, pd.Timestamp):
                    value = value.strftime("%Y-%m-%d")
                values.append(value)
            self.tree.insert("", "end", values=values)


@dataclass
class AppState:
    repo: DashboardRepository | None = None
    options: dict[str, list[dict[str, str]]] | None = None
    dashboard: dict[str, Any] | None = None
    evaluation_metrics: pd.DataFrame | None = None


class RetailDesktopApp(tk.Tk):
    def __init__(self, data_root: str | Path = DEFAULT_DATA_ROOT, artifacts_root: str | Path | None = None, title: str = DEFAULT_TITLE):
        super().__init__()
        self.title(title)
        self.geometry("1480x920")
        self.minsize(1260, 780)
        self.configure(bg="#eef3fb")

        self.state = AppState()
        self.data_root_var = tk.StringVar(value=str(Path(data_root)))
        self.artifacts_root_var = tk.StringVar(value=str(artifacts_root) if artifacts_root else "")
        self.store_var = tk.StringVar(value="Tất cả")
        self.category_var = tk.StringVar(value="Tất cả")
        self.product_var = tk.StringVar(value="Tất cả")
        self.holiday_var = tk.StringVar(value="Tất cả")
        self.period_var = tk.StringVar(value="30 ngày")
        self.model_var = tk.StringVar(value="ensemble")
        self.horizon_var = tk.IntVar(value=30)
        self.history_var = tk.IntVar(value=120)
        self.inventory_limit_var = tk.IntVar(value=12)

        self._store_lookup: dict[str, str] = {"Tất cả": "all"}
        self._category_lookup: dict[str, str] = {"Tất cả": "all"}
        self._product_lookup: dict[str, str] = {"Tất cả": "all"}
        self._holiday_lookup: dict[str, str] = {"Tất cả": "all"}
        self._period_lookup: dict[str, int] = {"7 ngày": 7, "30 ngày": 30, "1 quý": 90, "1 năm": 365, "Tùy chỉnh": 0}
        self._model_lookup: dict[str, str] = {}
        self._combos: dict[str, ttk.Combobox] = {}
        self._last_filters: dict[str, str | None] = {}
        self._decision_vars = {
            "status": tk.StringVar(value="-"),
            "current_stock": tk.StringVar(value="-"),
            "safety_stock": tk.StringVar(value="-"),
            "reorder_point": tk.StringVar(value="-"),
            "suggest_order": tk.StringVar(value="-"),
        }

        self._build_style()
        self._build_layout()
        self._set_status("Sẵn sàng. Bấm 'Tải dữ liệu'.")
        self.after(150, self.load_data)

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            preferred = "vista" if "vista" in style.theme_names() else ("xpnative" if "xpnative" in style.theme_names() else "clam")
            style.theme_use(preferred)
        except Exception:
            pass
        style.configure("App.TFrame", background="#eef3fb")
        style.configure("Panel.TFrame", background="white")
        style.configure("Banner.TFrame", background="#0f2d5c")
        style.configure("Banner.TLabel", background="#0f2d5c", foreground="white", font=("Segoe UI", 18, "bold"))
        style.configure("BannerSub.TLabel", background="#0f2d5c", foreground="#dbe8ff", font=("Segoe UI", 9))
        style.configure("Subtle.TLabel", background="#eef3fb", foreground="#51637f", font=("Segoe UI", 9))
        style.configure("Section.TLabel", background="#eef3fb", foreground="#163963", font=("Segoe UI", 12, "bold"))
        style.configure("CardValue.TLabel", background="white", foreground="#0f2d5c", font=("Segoe UI", 18, "bold"))
        style.configure("CardLabel.TLabel", background="white", foreground="#5f7aa8", font=("Segoe UI", 9))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("TNotebook", background="#eef3fb", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(12, 8), font=("Segoe UI", 10, "bold"))
        style.configure("Treeview", rowheight=26, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        style.configure("TCombobox", padding=(6, 4))
        style.configure("TSpinbox", padding=(6, 4))
        style.configure("TButton", padding=(10, 6))

    def _build_layout(self) -> None:
        banner = ttk.Frame(self, style="Banner.TFrame", padding=(18, 14))
        banner.pack(side="top", fill="x")
        ttk.Label(banner, text="Phần mềm dự báo bán lẻ", style="Banner.TLabel").pack(anchor="w")
        ttk.Label(
            banner,
            text="Giao diện desktop Python để dự báo nhu cầu, theo dõi tồn kho và xem mô hình trực tiếp từ thư mục kết quả.",
            style="BannerSub.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        body = ttk.Frame(self, style="App.TFrame", padding=12)
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, minsize=400)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        sidebar_scroll = ScrollableFrame(body, style="Panel.TFrame")
        sidebar_scroll.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        sidebar_scroll.grid_propagate(False)
        sidebar_scroll.configure(width=400)
        self.sidebar = ttk.Frame(sidebar_scroll.inner, style="Panel.TFrame", padding=16)
        self.sidebar.pack(fill="both", expand=True)
        self.content = ttk.Frame(body, style="App.TFrame")
        self.content.grid(row=0, column=1, sticky="nsew")

        self._build_sidebar()
        self._build_tabs()

        self.status_var = tk.StringVar(value="")
        status = ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(12, 6), style="Subtle.TLabel")
        status.pack(side="bottom", fill="x")

    def _build_sidebar(self) -> None:
        ttk.Label(self.sidebar, text="Nguồn dữ liệu", style="Section.TLabel").pack(anchor="w", pady=(0, 8))

        self._path_row(self.sidebar, "Thư mục dữ liệu", self.data_root_var, self._browse_data_root)
        self._path_row(self.sidebar, "Thư mục kết quả", self.artifacts_root_var, self._browse_artifacts_root)

        ttk.Separator(self.sidebar).pack(fill="x", pady=12)

        ttk.Label(self.sidebar, text="Bộ lọc", style="Section.TLabel").pack(anchor="w")
        self._combo_row(self.sidebar, "Kho", self.store_var, self._on_store_change, "store")
        self._combo_row(self.sidebar, "Nhóm sản phẩm", self.category_var, self._on_category_change, "category")
        self._combo_row(self.sidebar, "Sản phẩm", self.product_var, self._on_product_change, "product")
        self._combo_row(self.sidebar, "Ngày lễ", self.holiday_var, self._on_holiday_change, "holiday")
        self._combo_row(self.sidebar, "Mô hình", self.model_var, self._on_model_change, "model")
        self._combo_row(self.sidebar, "Khoảng xem nhanh", self.period_var, self._on_period_change, "period")

        ttk.Separator(self.sidebar).pack(fill="x", pady=12)
        self._spin_row(self.sidebar, "Số ngày dự báo", self.horizon_var, 1, 365)
        self._spin_row(self.sidebar, "Số ngày lịch sử", self.history_var, 30, 365)
        self._spin_row(self.sidebar, "Số dòng tồn kho hiển thị", self.inventory_limit_var, 1, 100)

        ttk.Separator(self.sidebar).pack(fill="x", pady=12)

        btn_row = ttk.Frame(self.sidebar, style="Panel.TFrame")
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Tải dữ liệu", command=self.load_data, style="Accent.TButton").pack(side="left", fill="x", expand=True)
        ttk.Button(btn_row, text="Áp dụng lọc", command=self.refresh_dashboard).pack(side="left", fill="x", expand=True, padx=(8, 0))

        ttk.Button(self.sidebar, text="Khôi phục mặc định", command=self._reset_filters).pack(fill="x", pady=(10, 0))

        self.sidebar_hint = ttk.Label(
            self.sidebar,
            text="Mẹo: Nếu app mở hơi lâu, cứ bấm Tải dữ liệu một lần rồi dùng tiếp. Lần sau sẽ nhanh hơn.",
            style="Subtle.TLabel",
            wraplength=280,
            justify="left",
        )
        self.sidebar_hint.pack(anchor="w", pady=(16, 0))

    def _path_row(self, master: tk.Misc, label: str, var: tk.StringVar, browse_cmd) -> None:
        frame = ttk.Frame(master, style="Panel.TFrame")
        frame.pack(fill="x", pady=(0, 8))
        ttk.Label(frame, text=label, style="Subtle.TLabel").pack(anchor="w")
        inner = ttk.Frame(frame, style="Panel.TFrame")
        inner.pack(fill="x", pady=(4, 0))
        entry = ttk.Entry(inner, textvariable=var)
        entry.pack(side="left", fill="x", expand=True)
        ttk.Button(inner, text="...", width=3, command=browse_cmd).pack(side="left", padx=(6, 0))

    def _combo_row(self, master: tk.Misc, label: str, var: tk.StringVar, on_change, attr_name: str) -> None:
        frame = ttk.Frame(master, style="Panel.TFrame")
        frame.pack(fill="x", pady=(0, 8))
        ttk.Label(frame, text=label, style="Subtle.TLabel").pack(anchor="w")
        combo = ttk.Combobox(frame, textvariable=var, state="readonly")
        combo.pack(fill="x", pady=(4, 0))
        combo.bind("<<ComboboxSelected>>", lambda _event: on_change())
        self._combos[attr_name] = combo

    def _spin_row(self, master: tk.Misc, label: str, var: tk.IntVar, from_: int, to: int) -> None:
        frame = ttk.Frame(master, style="Panel.TFrame")
        frame.pack(fill="x", pady=(0, 8))
        ttk.Label(frame, text=label, style="Subtle.TLabel").pack(anchor="w")
        spin = ttk.Spinbox(frame, from_=from_, to=to, textvariable=var, width=10)
        spin.pack(fill="x", pady=(4, 0))

    def _build_tabs(self) -> None:
        self.notebook = ttk.Notebook(self.content)
        self.notebook.pack(fill="both", expand=True)

        self.overview_tab = ttk.Frame(self.notebook, padding=12)
        self.forecast_tab = ttk.Frame(self.notebook, padding=12)
        self.inventory_tab = ttk.Frame(self.notebook, padding=12)
        self.models_tab = ttk.Frame(self.notebook, padding=12)
        self.data_tab = ttk.Frame(self.notebook, padding=12)

        self.notebook.add(self.overview_tab, text="Tổng quan")
        self.notebook.add(self.forecast_tab, text="Dự báo")
        self.notebook.add(self.inventory_tab, text="Tồn kho")
        self.notebook.add(self.models_tab, text="Mô hình")
        self.notebook.add(self.data_tab, text="Dữ liệu")

        self._build_overview_tab()
        self._build_forecast_tab()
        self._build_inventory_tab()
        self._build_models_tab()
        self._build_data_tab()

    def _build_overview_tab(self) -> None:
        self.overview_tab.grid_columnconfigure(0, weight=1)
        self.overview_tab.grid_columnconfigure(1, weight=1)
        self.overview_tab.grid_rowconfigure(2, weight=1)

        self.overview_cards = ttk.Frame(self.overview_tab)
        self.overview_cards.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        for idx in range(4):
            self.overview_cards.columnconfigure(idx, weight=1)

        self.overview_card_vars = {
            "total_units": tk.StringVar(value="0"),
            "total_revenue": tk.StringVar(value="0"),
            "forecast_units": tk.StringVar(value="0"),
            "stockout_rate": tk.StringVar(value="0%"),
        }

        self._build_card(self.overview_cards, 0, "Tổng thực tế", self.overview_card_vars["total_units"])
        self._build_card(self.overview_cards, 1, "Doanh thu", self.overview_card_vars["total_revenue"])
        self._build_card(self.overview_cards, 2, "Dự báo", self.overview_card_vars["forecast_units"])
        self._build_card(self.overview_cards, 3, "Tỷ lệ thiếu hàng", self.overview_card_vars["stockout_rate"])

        self.overview_chart = SeriesChart(self.overview_tab)
        self.overview_chart.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, 10))

        left = ttk.Frame(self.overview_tab)
        left.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        right = ttk.Frame(self.overview_tab)
        right.grid(row=2, column=1, sticky="nsew", padx=(8, 0))
        left.grid_rowconfigure(1, weight=1)
        right.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        ttk.Label(left, text="Top sản phẩm bán chạy", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
        self.top_products_table = DataTable(left, ["Sản phẩm", "Nhóm sản phẩm", "Thực tế", "Dự báo"], height=10)
        self.top_products_table.pack(fill="both", expand=True)

        ttk.Label(right, text="Tổng quan nhanh", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
        self.summary_text = tk.Text(right, height=14, wrap="word", bg="white", relief="solid", borderwidth=1, font=("Segoe UI", 10))
        self.summary_text.pack(fill="both", expand=True)

    def _build_forecast_tab(self) -> None:
        self.forecast_tab.grid_columnconfigure(0, weight=1)
        self.forecast_tab.grid_rowconfigure(3, weight=1)

        ttk.Label(self.forecast_tab, text="Dự báo theo sản phẩm", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        decision = ttk.Frame(self.forecast_tab, style="Panel.TFrame", padding=12)
        decision.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        for idx in range(4):
            decision.columnconfigure(idx, weight=1)
        self._build_card(decision, 0, "Trạng thái", self._decision_vars["status"])
        self._build_card(decision, 1, "Tồn kho hiện tại", self._decision_vars["current_stock"])
        self._build_card(decision, 2, "Tồn kho an toàn", self._decision_vars["safety_stock"])
        self._build_card(decision, 3, "Đề xuất đặt", self._decision_vars["suggest_order"])
        self.decision_note = tk.Text(
            self.forecast_tab,
            height=3,
            wrap="word",
            bg="white",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 10),
        )
        self.decision_note.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        split = ttk.Panedwindow(self.forecast_tab, orient="vertical")
        split.grid(row=3, column=0, sticky="nsew")
        chart_frame = ttk.Frame(split)
        table_frame = ttk.Frame(split)
        split.add(chart_frame, weight=3)
        split.add(table_frame, weight=2)

        self.forecast_chart = SeriesChart(chart_frame)
        self.forecast_chart.pack(fill="both", expand=True)
        self.forecast_table = DataTable(table_frame, ["Ngày", "Thực tế", "Dự báo"], height=8)
        self.forecast_table.pack(fill="both", expand=True)

    def _build_inventory_tab(self) -> None:
        self.inventory_tab.grid_columnconfigure(0, weight=1)
        self.inventory_tab.grid_rowconfigure(1, weight=0)
        self.inventory_tab.grid_rowconfigure(2, weight=1)

        stats = ttk.Frame(self.inventory_tab)
        stats.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for idx in range(4):
            stats.columnconfigure(idx, weight=1)

        self.inventory_card_vars = {
            "safety_stock_avg": tk.StringVar(value="0"),
            "reorder_point_avg": tk.StringVar(value="0"),
            "eoq_avg": tk.StringVar(value="0"),
            "coverage": tk.StringVar(value="0"),
        }
        self._build_card(stats, 0, "Tồn kho an toàn trung bình", self.inventory_card_vars["safety_stock_avg"])
        self._build_card(stats, 1, "Điểm đặt hàng lại trung bình", self.inventory_card_vars["reorder_point_avg"])
        self._build_card(stats, 2, "Lượng đặt hàng tối ưu trung bình", self.inventory_card_vars["eoq_avg"])
        self._build_card(stats, 3, "Mức phủ sóng", self.inventory_card_vars["coverage"])

        ttk.Label(self.inventory_tab, text="Bảng khuyến nghị tồn kho", style="Section.TLabel").grid(row=1, column=0, sticky="w", pady=(0, 6))
        self.inventory_table = DataTable(
            self.inventory_tab,
            ["Cửa hàng", "Sản phẩm", "Nhóm sản phẩm", "Nhu cầu trung bình", "Tồn kho an toàn", "Điểm đặt hàng lại", "Lượng đặt hàng tối ưu", "Mô hình"],
            height=14,
        )
        self.inventory_table.grid(row=2, column=0, sticky="nsew")

    def _build_models_tab(self) -> None:
        self.models_tab.grid_columnconfigure(0, weight=1)
        self.models_tab.grid_rowconfigure(2, weight=1)

        self.model_metrics_frame = ttk.Frame(self.models_tab)
        self.model_metrics_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for idx in range(4):
            self.model_metrics_frame.columnconfigure(idx, weight=1)

        self.model_card_vars = {
            "mae": tk.StringVar(value="0"),
            "rmse": tk.StringVar(value="0"),
            "mape": tk.StringVar(value="0%"),
            "bias": tk.StringVar(value="0%"),
        }
        self._build_card(self.model_metrics_frame, 0, "Sai số tuyệt đối TB", self.model_card_vars["mae"])
        self._build_card(self.model_metrics_frame, 1, "Căn sai số bình phương TB", self.model_card_vars["rmse"])
        self._build_card(self.model_metrics_frame, 2, "Sai số phần trăm TB", self.model_card_vars["mape"])
        self._build_card(self.model_metrics_frame, 3, "Độ lệch", self.model_card_vars["bias"])

        lower = ttk.Frame(self.models_tab)
        lower.grid(row=1, column=0, sticky="nsew")
        lower.grid_columnconfigure(0, weight=1)
        lower.grid_columnconfigure(1, weight=1)
        lower.grid_rowconfigure(1, weight=1)

        ttk.Label(lower, text="Mô hình dự báo", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Label(lower, text="Mức phủ sóng / số nhóm", style="Section.TLabel").grid(row=0, column=1, sticky="w", pady=(0, 6))

        self.model_metrics_table = DataTable(lower, ["Nhóm", "Mô hình", "Sai số tuyệt đối TB", "Căn sai số bình phương TB", "Sai số phần trăm TB", "Độ lệch", "Dự báo trung bình", "Số nhóm phủ"], height=12)
        self.model_metrics_table.grid(row=1, column=0, sticky="nsew", padx=(0, 6))

        self.model_coverage_table = DataTable(lower, ["Mô hình", "Số dòng", "Số nhóm", "Ngày dự báo đầu", "Ngày dự báo cuối"], height=12)
        self.model_coverage_table.grid(row=1, column=1, sticky="nsew", padx=(6, 0))

    def _build_data_tab(self) -> None:
        self.data_tab.grid_columnconfigure(0, weight=1)
        self.data_tab.grid_rowconfigure(1, weight=1)
        self.data_tab.grid_rowconfigure(2, weight=1)

        self.data_summary = tk.Text(self.data_tab, height=8, wrap="word", bg="white", relief="solid", borderwidth=1, font=("Segoe UI", 10))
        self.data_summary.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        lower = ttk.Frame(self.data_tab)
        lower.grid(row=1, column=0, sticky="nsew")
        lower.grid_columnconfigure(0, weight=1)
        lower.grid_columnconfigure(1, weight=1)
        lower.grid_rowconfigure(0, weight=1)

        ttk.Label(lower, text="Dữ liệu đầu", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Label(lower, text="Tồn kho ưu tiên", style="Section.TLabel").grid(row=0, column=1, sticky="w", pady=(0, 6))

        self.history_preview = DataTable(lower, ["Ngày", "Mã cửa hàng", "Mã sản phẩm", "Thực tế", "Doanh thu"], height=11)
        self.history_preview.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.inventory_preview = DataTable(lower, ["Cửa hàng", "Sản phẩm", "Tồn kho an toàn", "Điểm đặt hàng lại", "Lượng đặt hàng tối ưu"], height=11)
        self.inventory_preview.grid(row=1, column=1, sticky="nsew", padx=(6, 0))

    def _build_card(self, master: tk.Misc, column: int, label: str, var: tk.StringVar) -> None:
        card = ttk.Frame(master, style="Panel.TFrame", padding=14)
        card.grid(row=0, column=column, sticky="nsew", padx=6)
        ttk.Label(card, text=label, style="CardLabel.TLabel").pack(anchor="w")
        ttk.Label(card, textvariable=var, style="CardValue.TLabel").pack(anchor="w", pady=(8, 0))

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.update_idletasks()

    def _browse_data_root(self) -> None:
        folder = filedialog.askdirectory(title="Chọn thư mục dữ liệu")
        if folder:
            self.data_root_var.set(folder)

    def _browse_artifacts_root(self) -> None:
        folder = filedialog.askdirectory(title="Chọn thư mục kết quả")
        if folder:
            self.artifacts_root_var.set(folder)

    def _reset_filters(self) -> None:
        self.store_var.set("Tất cả")
        self.category_var.set("Tất cả")
        self.product_var.set("Tất cả")
        self.model_var.set("ensemble")
        self.horizon_var.set(30)
        self.history_var.set(120)
        self.inventory_limit_var.set(12)
        self.refresh_dashboard()

    def _selected_value(self, lookup: dict[str, str], current: str) -> str | None:
        value = lookup.get(current)
        if not value or value == "all":
            return None
        return value

    def load_data(self) -> None:
        data_root = Path(self.data_root_var.get().strip() or DEFAULT_DATA_ROOT)
        artifacts_text = self.artifacts_root_var.get().strip()
        artifacts_root = Path(artifacts_text) if artifacts_text else None
        self._set_status("Đang tải dữ liệu...")
        try:
            repo = DashboardRepository(data_root, artifacts_root)
            options = repo.options()
            evaluation = _load_evaluation_metrics([p for p in [artifacts_root, data_root] if p is not None])
        except Exception as exc:
            messagebox.showerror("Lỗi tải dữ liệu", str(exc))
            self._set_status("Tải dữ liệu thất bại.")
            return

        self.state.repo = repo
        self.state.options = options
        self.state.evaluation_metrics = evaluation
        self._populate_filters(options)
        self._set_status(f"Đã tải dữ liệu từ {data_root}.")
        self.refresh_dashboard()

    def _populate_filters(self, options: dict[str, list[dict[str, str]]]) -> None:
        stores = {"Tất cả": "all"}
        for opt in options.get("stores", []):
            stores[opt["label"]] = opt["value"]
        categories = {"Tất cả": "all"}
        for opt in options.get("categories", []):
            categories[opt["label"]] = opt["value"]
        holidays = {"Tất cả": "all"}
        for opt in options.get("holidays", []):
            holidays[opt["label"]] = opt["value"]
        models: dict[str, str] = {}
        for opt in options.get("models", []):
            models[opt["label"]] = opt["value"]

        self._store_lookup = stores
        self._category_lookup = categories
        self._holiday_lookup = holidays
        self._model_lookup = models or {"Ensemble": "ensemble", "Baseline": "baseline"}

        self._combos["store"]["values"] = list(stores.keys())
        self._combos["category"]["values"] = list(categories.keys())
        self._combos["holiday"]["values"] = list(holidays.keys())
        self._combos["period"]["values"] = list(self._period_lookup.keys())
        self._combos["model"]["values"] = list(models.keys()) or ["Ensemble", "Baseline"]

        if self.store_var.get() not in stores:
            self.store_var.set("Tất cả")
        if self.category_var.get() not in categories:
            self.category_var.set("Tất cả")
        if self.holiday_var.get() not in holidays:
            self.holiday_var.set("Tất cả")
        if self.period_var.get() not in self._period_lookup:
            self.period_var.set("30 ngày")
        if self.model_var.get() not in self._model_lookup:
            self.model_var.set(next(iter(self._model_lookup.keys())))

        self._refresh_product_options()

    def _on_store_change(self) -> None:
        self._refresh_product_options()
        self.refresh_dashboard()

    def _on_category_change(self) -> None:
        self._refresh_product_options()
        self.refresh_dashboard()

    def _on_product_change(self) -> None:
        self.refresh_dashboard()

    def _on_holiday_change(self) -> None:
        self.refresh_dashboard()

    def _on_model_change(self) -> None:
        self.refresh_dashboard()

    def _on_period_change(self) -> None:
        selection = self.period_var.get()
        span = self._period_lookup.get(selection, 0)
        if span:
            self.horizon_var.set(span)
            if span <= 30:
                self.history_var.set(max(90, span * 3))
            elif span <= 90:
                self.history_var.set(max(180, span * 2))
            else:
                self.history_var.set(max(365, span))
            self.refresh_dashboard()

    def _resolve_filters(self) -> dict[str, str | None]:
        return {
            "store_id": self._selected_value(self._store_lookup, self.store_var.get()),
            "item_id": self._selected_value(self._product_lookup, self.product_var.get()),
            "category": self._selected_value(self._category_lookup, self.category_var.get()),
            "holiday_name": self._selected_value(self._holiday_lookup, self.holiday_var.get()),
            "model_name": self._model_lookup.get(self.model_var.get(), self.model_var.get() or "ensemble"),
        }

    def _render_decision_panel(self, dashboard: dict[str, Any], filters: dict[str, str | None]) -> None:
        inventory = pd.DataFrame(dashboard.get("inventory", []))
        history = self.state.repo.history if self.state.repo is not None else pd.DataFrame()

        if inventory.empty:
            self._decision_vars["status"].set("-")
            self._decision_vars["current_stock"].set("-")
            self._decision_vars["safety_stock"].set("-")
            self._decision_vars["reorder_point"].set("-")
            self._decision_vars["suggest_order"].set("-")
            self.decision_note.delete("1.0", "end")
            self.decision_note.insert("end", "Hãy chọn Kho / Nhóm sản phẩm / Sản phẩm / Ngày lễ để hệ thống đưa ra gợi ý đặt hàng.")
            return

        row = inventory.iloc[0]
        safety_stock = float(row.get("Safety Stock") or row.get("safety_stock") or 0.0)
        reorder_point = float(row.get("Reorder Point") or row.get("reorder_point") or 0.0)
        eoq = float(row.get("EOQ") or row.get("eoq") or 0.0)

        current_stock = 0.0
        if not history.empty and "stock_end" in history.columns:
            scope = history.copy()
            if filters.get("store_id") and "store_id" in scope.columns:
                scope = scope[scope["store_id"].astype(str) == str(filters["store_id"])]
            if filters.get("item_id") and "item_id" in scope.columns:
                scope = scope[scope["item_id"].astype(str) == str(filters["item_id"])]
            if filters.get("category") and "category" in scope.columns:
                scope = scope[scope["category"].astype(str) == str(filters["category"])]
            if not scope.empty:
                scope = scope.copy()
                scope["date"] = pd.to_datetime(scope["date"], errors="coerce")
                scope = scope.dropna(subset=["date"]).sort_values("date")
                latest_date = scope["date"].max()
                latest = scope[scope["date"] == latest_date]
                if "stock_end" in latest.columns:
                    current_stock = float(pd.to_numeric(latest["stock_end"], errors="coerce").fillna(0).sum())

        if current_stock >= reorder_point:
            status = "Đủ hàng"
        elif current_stock >= safety_stock:
            status = "Cần nhập"
        else:
            status = "Thiếu hàng"

        suggest_order = max(reorder_point - current_stock, 0.0)
        if suggest_order <= 0 and status == "Đủ hàng":
            suggest_order = 0.0
        elif suggest_order <= 0:
            suggest_order = eoq if eoq > 0 else 0.0

        self._decision_vars["status"].set(status)
        self._decision_vars["current_stock"].set(_compact_number(current_stock))
        self._decision_vars["safety_stock"].set(_compact_number(safety_stock))
        self._decision_vars["reorder_point"].set(_compact_number(reorder_point))
        self._decision_vars["suggest_order"].set(_compact_number(suggest_order))

        lines = [
            f"Trạng thái hiện tại: {status}",
            f"Tồn kho hiện tại: {_compact_number(current_stock)}",
            f"Tồn kho an toàn / Điểm đặt hàng lại: {_compact_number(safety_stock)} / {_compact_number(reorder_point)}",
            f"Đề xuất đặt thêm: {_compact_number(suggest_order)}",
        ]
        if filters.get("holiday_name"):
            lines.append(f"Đang xem riêng theo: {filters['holiday_name']}")
        if status == "Thiếu hàng":
            lines.append("Khuyến nghị: đặt ngay để tránh thiếu hàng.")
        elif status == "Cần nhập":
            lines.append("Khuyến nghị: chuẩn bị nhập trong ngắn hạn.")
        else:
            lines.append("Khuyến nghị: tạm đủ, theo dõi thêm nhu cầu.")
        self.decision_note.delete("1.0", "end")
        self.decision_note.insert("end", "\n".join(lines))

    def _refresh_product_options(self) -> None:
        if self.state.repo is None or "product" not in self._combos:
            return

        history = self.state.repo.history.copy()
        products = self.state.repo.products.copy()
        if products.empty and not history.empty:
            products = history[[c for c in ["item_id", "product_name", "category"] if c in history.columns]].drop_duplicates()

        if products.empty:
            product_map = {"Tất cả": "all"}
        else:
            store_id = self._selected_value(self._store_lookup, self.store_var.get())
            category = self._selected_value(self._category_lookup, self.category_var.get())
            if store_id and "store_id" in history.columns and "item_id" in history.columns:
                allowed_items = history.loc[history["store_id"].astype(str) == str(store_id), "item_id"].dropna().astype(str).unique().tolist()
                if "item_id" in products.columns:
                    products = products[products["item_id"].astype(str).isin(allowed_items)]
            if category and "category" in products.columns:
                products = products[products["category"].astype(str) == str(category)]
            if products.empty:
                product_map = {"Tất cả": "all"}
            else:
                if "product_name" not in products.columns:
                    products = products.assign(product_name=products["item_id"].astype(str))
                product_map = {"Tất cả": "all"}
                for _, row in products.sort_values([c for c in ["category", "product_name"] if c in products.columns]).drop_duplicates("item_id").iterrows():
                    label = _nice_product_label(row)
                    product_map[label] = str(row.get("item_id"))

        current_value = self._selected_value(self._product_lookup, self.product_var.get())
        self._product_lookup = product_map
        self._combos["product"]["values"] = list(product_map.keys())
        if current_value and current_value in product_map.values():
            for label, value in product_map.items():
                if value == current_value:
                    self.product_var.set(label)
                    break
        else:
            self.product_var.set("Tất cả")

    def refresh_dashboard(self) -> None:
        if self.state.repo is None:
            self._clear_views()
            return

        filters = self._resolve_filters()
        self._last_filters = filters
        horizon = int(self.horizon_var.get() or 30)
        history_window = int(self.history_var.get() or 120)
        inventory_limit = int(self.inventory_limit_var.get() or 12)

        self._set_status("Đang dựng dashboard...")
        try:
            dashboard = self.state.repo.dashboard(
                store_id=filters["store_id"],
                item_id=filters["item_id"],
                category=filters["category"],
                holiday_name=filters["holiday_name"],
                model_name=str(filters["model_name"]),
                horizon=horizon,
                history_window=history_window,
                inventory_limit=inventory_limit,
            )
            evaluation = self.state.evaluation_metrics if self.state.evaluation_metrics is not None else _load_evaluation_metrics(self.state.repo.roots)
        except Exception as exc:
            messagebox.showerror("Lỗi hiển thị", str(exc))
            self._set_status("Không thể dựng dashboard.")
            return

        self.state.dashboard = dashboard
        self.state.evaluation_metrics = evaluation
        self._render_overview(dashboard)
        self._render_decision_panel(dashboard, filters)
        self._render_forecast(dashboard)
        self._render_inventory(dashboard)
        self._render_models(dashboard, evaluation)
        self._render_data_tab(dashboard, evaluation)
        self._set_status("Đã cập nhật xong.")

    def _clear_views(self) -> None:
        for var in self.overview_card_vars.values():
            var.set("0")
        for var in self.inventory_card_vars.values():
            var.set("0")
        for var in self.model_card_vars.values():
            var.set("0")
        self.summary_text.delete("1.0", "end")
        self.data_summary.delete("1.0", "end")
        for table in [
            self.top_products_table,
            self.forecast_table,
            self.inventory_table,
            self.model_metrics_table,
            self.model_coverage_table,
            self.history_preview,
            self.inventory_preview,
        ]:
            table.set_rows([])
        empty = pd.DataFrame()
        self.overview_chart.draw(empty, empty, "Biểu đồ tổng quan")
        self.forecast_chart.draw(empty, empty, "Biểu đồ dự báo")

    def _render_overview(self, dashboard: dict[str, Any]) -> None:
        summary = dashboard.get("summary", {})
        self.overview_card_vars["total_units"].set(_compact_number(summary.get("total_units")))
        self.overview_card_vars["total_revenue"].set(_compact_number(summary.get("total_revenue")))
        self.overview_card_vars["forecast_units"].set(_compact_number(summary.get("forecast_units")))
        self.overview_card_vars["stockout_rate"].set(f"{float(summary.get('stockout_rate') or 0.0):.1f}%")

        actual = pd.DataFrame(dashboard.get("series", {}).get("actual", []))
        forecast = pd.DataFrame(dashboard.get("series", {}).get("forecast", []))
        if not actual.empty and "value" not in actual.columns:
            actual = actual.rename(columns={"target": "value"})
        if not forecast.empty and "value" not in forecast.columns:
            forecast = forecast.rename(columns={"forecast": "value"})

        self.overview_chart.draw(actual, forecast, "So sánh thực tế và dự báo")

        top_products = pd.DataFrame(dashboard.get("top_products", []))
        if not top_products.empty:
            cols = {
                "product_name": "Sản phẩm",
                "category": "Nhóm sản phẩm",
                "actual_units": "Thực tế",
                "forecast_units": "Dự báo",
            }
            top_products = top_products.rename(columns=cols)
            for col in ["Thực tế", "Dự báo"]:
                if col in top_products.columns:
                    top_products[col] = top_products[col].apply(_compact_number)
            wanted = [col for col in ["Sản phẩm", "Nhóm sản phẩm", "Thực tế", "Dự báo"] if col in top_products.columns]
            self.top_products_table.set_rows(top_products[wanted], columns=wanted)
        else:
            self.top_products_table.set_rows([])

        lines = [
            f"Khoảng dữ liệu thực tế: {summary.get('actual_start', '-') } → {summary.get('actual_end', '-')}",
            f"Dự báo đến: {summary.get('forecast_end', '-')}",
            f"Sản phẩm: {summary.get('product_count', 0)}",
            f"Cửa hàng: {summary.get('store_count', 0)}",
            f"Nhóm sản phẩm: {summary.get('category_count', 0)}",
            f"Mức phủ sóng: {_compact_number(summary.get('coverage'))}",
            f"Tỷ lệ dự báo: {float(summary.get('forecast_rate') or 0.0):.1f}%",
        ]
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("end", "\n".join(lines))

    def _render_forecast(self, dashboard: dict[str, Any]) -> None:
        series = dashboard.get("series", {})
        actual = pd.DataFrame(series.get("actual", []))
        forecast = pd.DataFrame(series.get("forecast", []))
        if not actual.empty and "value" not in actual.columns:
            actual = actual.rename(columns={"target": "value"})
        if not forecast.empty and "value" not in forecast.columns:
            forecast = forecast.rename(columns={"forecast": "value"})
        self.forecast_chart.draw(actual, forecast, "Dự báo theo ngày")

        rows = []
        if not actual.empty:
            actual_copy = actual.copy()
            actual_copy["Ngày"] = pd.to_datetime(actual_copy["date"], errors="coerce")
            actual_copy["Ngày"] = actual_copy["Ngày"].dt.strftime("%Y-%m-%d")
            actual_copy["Thực tế"] = actual_copy["value"]
            actual_copy["Dự báo"] = None
            rows.append(actual_copy[["Ngày", "Thực tế", "Dự báo"]])
        if not forecast.empty:
            forecast_copy = forecast.copy()
            forecast_copy["Ngày"] = pd.to_datetime(forecast_copy["date"], errors="coerce")
            forecast_copy["Ngày"] = forecast_copy["Ngày"].dt.strftime("%Y-%m-%d")
            forecast_copy["Thực tế"] = None
            forecast_copy["Dự báo"] = forecast_copy["value"]
            rows.append(forecast_copy[["Ngày", "Thực tế", "Dự báo"]])
        table = pd.concat([frame for frame in rows if not frame.empty], ignore_index=True) if rows else pd.DataFrame()
        if not table.empty:
            table["Thực tế"] = table["Thực tế"].apply(lambda v: _compact_number(v) if pd.notna(v) else "")
            table["Dự báo"] = table["Dự báo"].apply(lambda v: _compact_number(v) if pd.notna(v) else "")
            self.forecast_table.set_rows(table.tail(120), columns=["Ngày", "Thực tế", "Dự báo"])
        else:
            self.forecast_table.set_rows([])

    def _render_inventory(self, dashboard: dict[str, Any]) -> None:
        summary = dashboard.get("summary", {})
        self.inventory_card_vars["safety_stock_avg"].set(_compact_number(summary.get("safety_stock_avg")))
        self.inventory_card_vars["reorder_point_avg"].set(_compact_number(summary.get("reorder_point_avg")))
        self.inventory_card_vars["eoq_avg"].set(_compact_number(summary.get("eoq_avg")))
        self.inventory_card_vars["coverage"].set(_compact_number(summary.get("coverage")))

        inventory = pd.DataFrame(dashboard.get("inventory", []))
        if not inventory.empty:
            cols = {
                "store_name": "Cửa hàng",
                "product_name": "Sản phẩm",
                "category": "Nhóm sản phẩm",
                "avg_demand": "Nhu cầu trung bình",
                "safety_stock": "Tồn kho an toàn",
                "reorder_point": "Điểm đặt hàng lại",
                "eoq": "Lượng đặt hàng tối ưu",
                "source_model": "Mô hình",
            }
            inventory = inventory.rename(columns=cols)
            for col in ["Nhu cầu trung bình", "Tồn kho an toàn", "Điểm đặt hàng lại", "Lượng đặt hàng tối ưu"]:
                if col in inventory.columns:
                    inventory[col] = inventory[col].apply(_compact_number)
            wanted = [col for col in ["Cửa hàng", "Sản phẩm", "Nhóm sản phẩm", "Nhu cầu trung bình", "Tồn kho an toàn", "Điểm đặt hàng lại", "Lượng đặt hàng tối ưu", "Mô hình"] if col in inventory.columns]
            self.inventory_table.set_rows(inventory[wanted], columns=wanted)
            preview_cols = [col for col in ["Cửa hàng", "Sản phẩm", "Tồn kho an toàn", "Điểm đặt hàng lại", "Lượng đặt hàng tối ưu"] if col in inventory.columns]
            self.inventory_preview.set_rows(inventory[preview_cols].head(20), columns=preview_cols)
        else:
            self.inventory_table.set_rows([])
            self.inventory_preview.set_rows([])

    def _render_models(self, dashboard: dict[str, Any], evaluation: pd.DataFrame) -> None:
        regression = pd.DataFrame()
        forecast_models = pd.DataFrame()
        if evaluation is not None and not evaluation.empty:
            if "section" in evaluation.columns:
                regression = evaluation[evaluation["section"].astype(str) == "regression"].copy()
                forecast_models = evaluation[evaluation["section"].astype(str) == "forecast_models"].copy()
            else:
                forecast_models = evaluation.copy()

        if not regression.empty:
            row = regression.iloc[0]
            self.model_card_vars["mae"].set(_compact_number(row.get("mae")))
            self.model_card_vars["rmse"].set(_compact_number(row.get("rmse")))
            self.model_card_vars["mape"].set(_format_percent(row.get("mape")))
            self.model_card_vars["bias"].set(f"{float(row.get('bias') or 0.0):.2f}%")
        else:
            self.model_card_vars["mae"].set("0")
            self.model_card_vars["rmse"].set("0")
            self.model_card_vars["mape"].set("0%")
            self.model_card_vars["bias"].set("0%")

        if not evaluation.empty:
            display = evaluation.copy()
            rename_cols = {
                "section": "Nhóm",
                "model_name": "Mô hình",
                "mae": "Sai số tuyệt đối TB",
                "rmse": "Căn sai số bình phương TB",
                "mape": "Sai số phần trăm TB",
                "bias": "Độ lệch",
                "mean_forecast": "Dự báo trung bình",
                "coverage_groups": "Số nhóm phủ",
            }
            display = display.rename(columns=rename_cols)
            for col in ["Sai số tuyệt đối TB", "Căn sai số bình phương TB", "Sai số phần trăm TB", "Độ lệch", "Dự báo trung bình", "Số nhóm phủ"]:
                if col in display.columns:
                    if col in {"Sai số phần trăm TB", "Độ lệch"}:
                        display[col] = display[col].apply(lambda v: _format_percent(v) if pd.notna(v) else "")
                    else:
                        display[col] = display[col].apply(lambda v: _compact_number(v) if pd.notna(v) else "")
            wanted = [c for c in ["Nhóm", "Mô hình", "Sai số tuyệt đối TB", "Căn sai số bình phương TB", "Sai số phần trăm TB", "Độ lệch", "Dự báo trung bình", "Số nhóm phủ"] if c in display.columns]
            self.model_metrics_table.set_rows(display[wanted], columns=wanted)
        else:
            self.model_metrics_table.set_rows([])

        metrics = pd.DataFrame(dashboard.get("metrics", []))
        if not metrics.empty:
            display = metrics.rename(columns={
                "model_name": "Mô hình",
                "rows": "Số dòng",
                "groups": "Số nhóm",
                "min_forecast_date": "Ngày dự báo đầu",
                "max_forecast_date": "Ngày dự báo cuối",
            })
            for col in ["Số dòng", "Số nhóm"]:
                if col in display.columns:
                    display[col] = display[col].apply(lambda v: _compact_number(v) if pd.notna(v) else "")
            for col in ["Ngày dự báo đầu", "Ngày dự báo cuối"]:
                if col in display.columns:
                    display[col] = display[col].apply(_human_date)
            wanted = [col for col in ["Mô hình", "Số dòng", "Số nhóm", "Ngày dự báo đầu", "Ngày dự báo cuối"] if col in display.columns]
            self.model_coverage_table.set_rows(display[wanted], columns=wanted)
        else:
            self.model_coverage_table.set_rows([])

    def _render_data_tab(self, dashboard: dict[str, Any], evaluation: pd.DataFrame) -> None:
        summary = dashboard.get("summary", {})
        text = [
            f"Thư mục dữ liệu: {self.data_root_var.get()}",
            f"Thư mục kết quả: {self.artifacts_root_var.get() or '(không đặt)'}",
            "",
            f"Số sản phẩm: {summary.get('product_count', 0)}",
            f"Số cửa hàng: {summary.get('store_count', 0)}",
            f"Số nhóm sản phẩm: {summary.get('category_count', 0)}",
            f"Thực tế: {_compact_number(summary.get('total_units'))}",
            f"Dự báo: {_compact_number(summary.get('forecast_units'))}",
        ]
        if evaluation is not None and not evaluation.empty:
            text.extend(
                [
                    "",
                    "Các mô hình đã đọc:",
                    ", ".join(sorted(evaluation["Mô hình"].dropna().astype(str).unique().tolist())) if "Mô hình" in evaluation.columns else ", ".join(sorted(evaluation["model_name"].dropna().astype(str).unique().tolist())),
                ]
            )
        self.data_summary.delete("1.0", "end")
        self.data_summary.insert("end", "\n".join(text))

        history = pd.DataFrame(self.state.repo.history.head(25).to_dict(orient="records")) if self.state.repo is not None else pd.DataFrame()
        if not history.empty:
            history = history.rename(columns={
                "date": "Ngày",
                "store_id": "Mã cửa hàng",
                "item_id": "Mã sản phẩm",
                "target": "Thực tế",
                "revenue": "Doanh thu",
            })
            wanted = [col for col in ["Ngày", "Mã cửa hàng", "Mã sản phẩm", "Thực tế", "Doanh thu"] if col in history.columns]
            if wanted:
                history_preview = history[wanted].copy()
                if "Ngày" in history_preview.columns:
                    history_preview["Ngày"] = pd.to_datetime(history_preview["Ngày"], errors="coerce").dt.strftime("%Y-%m-%d")
                for col in ["Thực tế", "Doanh thu"]:
                    if col in history_preview.columns:
                        history_preview[col] = history_preview[col].apply(_compact_number)
                self.history_preview.set_rows(history_preview, columns=wanted)
        else:
            self.history_preview.set_rows([])

        if self.state.repo is not None:
            inventory_preview = pd.DataFrame(dashboard.get("inventory", []))
            if not inventory_preview.empty:
                wanted_inv = [col for col in ["Cửa hàng", "Sản phẩm", "Tồn kho an toàn", "Điểm đặt hàng lại", "Lượng đặt hàng tối ưu"] if col in inventory_preview.columns]
                self.inventory_preview.set_rows(inventory_preview[wanted_inv].head(20), columns=wanted_inv)
            else:
                self.inventory_preview.set_rows([])


def run_desktop_app(
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    artifacts_root: str | Path | None = None,
    title: str = DEFAULT_TITLE,
) -> None:
    app = RetailDesktopApp(data_root=data_root, artifacts_root=artifacts_root, title=title)
    app.mainloop()


if __name__ == "__main__":  # pragma: no cover
    run_desktop_app()

