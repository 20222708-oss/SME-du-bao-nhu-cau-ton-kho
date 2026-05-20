from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date as date_cls
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


VN_WEEKDAY_NAMES = {
    0: "Thứ Hai",
    1: "Thứ Ba",
    2: "Thứ Tư",
    3: "Thứ Năm",
    4: "Thứ Sáu",
    5: "Thứ Bảy",
    6: "Chủ Nhật",
}

VN_MONTH_NAMES = {
    1: "Tháng 1",
    2: "Tháng 2",
    3: "Tháng 3",
    4: "Tháng 4",
    5: "Tháng 5",
    6: "Tháng 6",
    7: "Tháng 7",
    8: "Tháng 8",
    9: "Tháng 9",
    10: "Tháng 10",
    11: "Tháng 11",
    12: "Tháng 12",
}

HOLIDAY_PRIORITY = {
    "Tết Nguyên Đán": 100,
    "Tết Dương lịch": 90,
    "Black Friday": 80,
    "Noel": 70,
    "Trung Thu": 65,
    "Quốc khánh": 60,
    "Giải phóng miền Nam": 55,
    "Quốc tế Lao động": 54,
    "Quốc tế Phụ nữ": 53,
    "Phụ nữ Việt Nam": 52,
    "Back to School": 45,
}


@dataclass(frozen=True)
class SyntheticConfig:
    output_dir: Path
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    num_products: int = 100
    num_stores: int = 5
    num_suppliers: int = 8
    seed: int = 42


@dataclass(frozen=True)
class HolidayEvent:
    start: pd.Timestamp
    end: pd.Timestamp
    name: str
    before_days: int = 7
    after_days: int = 3
    priority: int = 50


def _slug(text: str) -> str:
    return (
        text.lower()
        .replace("đ", "d")
        .replace("á", "a")
        .replace("à", "a")
        .replace("ả", "a")
        .replace("ã", "a")
        .replace("ạ", "a")
        .replace("ă", "a")
        .replace("ắ", "a")
        .replace("ằ", "a")
        .replace("ẳ", "a")
        .replace("ẵ", "a")
        .replace("ặ", "a")
        .replace("â", "a")
        .replace("ấ", "a")
        .replace("ầ", "a")
        .replace("ẩ", "a")
        .replace("ẫ", "a")
        .replace("ậ", "a")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ẻ", "e")
        .replace("ẽ", "e")
        .replace("ẹ", "e")
        .replace("ê", "e")
        .replace("ế", "e")
        .replace("ề", "e")
        .replace("ể", "e")
        .replace("ễ", "e")
        .replace("ệ", "e")
        .replace("í", "i")
        .replace("ì", "i")
        .replace("ỉ", "i")
        .replace("ĩ", "i")
        .replace("ị", "i")
        .replace("ó", "o")
        .replace("ò", "o")
        .replace("ỏ", "o")
        .replace("õ", "o")
        .replace("ọ", "o")
        .replace("ô", "o")
        .replace("ố", "o")
        .replace("ồ", "o")
        .replace("ổ", "o")
        .replace("ỗ", "o")
        .replace("ộ", "o")
        .replace("ơ", "o")
        .replace("ớ", "o")
        .replace("ờ", "o")
        .replace("ở", "o")
        .replace("ỡ", "o")
        .replace("ợ", "o")
        .replace("ú", "u")
        .replace("ù", "u")
        .replace("ủ", "u")
        .replace("ũ", "u")
        .replace("ụ", "u")
        .replace("ư", "u")
        .replace("ứ", "u")
        .replace("ừ", "u")
        .replace("ử", "u")
        .replace("ữ", "u")
        .replace("ự", "u")
        .replace("ý", "y")
        .replace("ỳ", "y")
        .replace("ỷ", "y")
        .replace("ỹ", "y")
        .replace("ỵ", "y")
    )


def _month_name(month: int) -> str:
    return VN_MONTH_NAMES.get(month, f"Tháng {month}")


def _weekday_name(dayofweek: int) -> str:
    return VN_WEEKDAY_NAMES[dayofweek]


def _season_for_date(ts: pd.Timestamp) -> str:
    if ts.month in {1, 2}:
        return "Đông"
    if ts.month in {3, 4, 5}:
        return "Xuân"
    if ts.month in {6, 7, 8}:
        return "Hè"
    if ts.month in {9, 10, 11}:
        return "Thu"
    return "Cuối năm"


def _fourth_friday_of_november(year: int) -> pd.Timestamp:
    nov1 = pd.Timestamp(year=year, month=11, day=1)
    offset = (4 - nov1.dayofweek) % 7
    first_friday = nov1 + pd.Timedelta(days=offset)
    return first_friday + pd.Timedelta(days=21)


def _holiday_events_for_year(year: int) -> list[HolidayEvent]:
    tet_windows = {
        2022: (pd.Timestamp("2022-01-29"), pd.Timestamp("2022-02-06")),
        2023: (pd.Timestamp("2023-01-20"), pd.Timestamp("2023-01-27")),
        2024: (pd.Timestamp("2024-02-08"), pd.Timestamp("2024-02-14")),
        2025: (pd.Timestamp("2025-01-27"), pd.Timestamp("2025-02-02")),
    }
    tet_start, tet_end = tet_windows.get(year, (pd.Timestamp(f"{year}-01-25"), pd.Timestamp(f"{year}-01-31")))
    events = [
        HolidayEvent(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-01-01"), "Tết Dương lịch", 4, 2, HOLIDAY_PRIORITY["Tết Dương lịch"]),
        HolidayEvent(pd.Timestamp(f"{year}-03-08"), pd.Timestamp(f"{year}-03-08"), "Quốc tế Phụ nữ", 3, 1, HOLIDAY_PRIORITY["Quốc tế Phụ nữ"]),
        HolidayEvent(pd.Timestamp(f"{year}-04-30"), pd.Timestamp(f"{year}-04-30"), "Giải phóng miền Nam", 3, 1, HOLIDAY_PRIORITY["Giải phóng miền Nam"]),
        HolidayEvent(pd.Timestamp(f"{year}-05-01"), pd.Timestamp(f"{year}-05-01"), "Quốc tế Lao động", 3, 1, HOLIDAY_PRIORITY["Quốc tế Lao động"]),
        HolidayEvent(pd.Timestamp(f"{year}-09-02"), pd.Timestamp(f"{year}-09-02"), "Quốc khánh", 4, 2, HOLIDAY_PRIORITY["Quốc khánh"]),
        HolidayEvent(pd.Timestamp(f"{year}-10-20"), pd.Timestamp(f"{year}-10-20"), "Phụ nữ Việt Nam", 3, 1, HOLIDAY_PRIORITY["Phụ nữ Việt Nam"]),
        HolidayEvent(pd.Timestamp(f"{year}-12-24"), pd.Timestamp(f"{year}-12-25"), "Noel", 5, 2, HOLIDAY_PRIORITY["Noel"]),
        HolidayEvent(_fourth_friday_of_november(year), _fourth_friday_of_november(year), "Black Friday", 4, 2, HOLIDAY_PRIORITY["Black Friday"]),
        HolidayEvent(pd.Timestamp(f"{year}-09-15"), pd.Timestamp(f"{year}-09-30"), "Trung Thu", 3, 2, HOLIDAY_PRIORITY["Trung Thu"]),
        HolidayEvent(tet_start, tet_end, "Tết Nguyên Đán", 14, 7, HOLIDAY_PRIORITY["Tết Nguyên Đán"]),
        HolidayEvent(pd.Timestamp(f"{year}-08-15"), pd.Timestamp(f"{year}-09-15"), "Back to School", 2, 1, HOLIDAY_PRIORITY["Back to School"]),
    ]
    return events


def _calendar_event_for_date(ts: pd.Timestamp) -> tuple[str, bool, bool]:
    events = _holiday_events_for_year(ts.year)
    active_name = ""
    best_priority = -1
    before_flag = False
    after_flag = False
    for event in events:
        if event.start <= ts <= event.end:
            if event.priority > best_priority:
                active_name = event.name
                best_priority = event.priority
        if event.start - pd.Timedelta(days=event.before_days) <= ts < event.start:
            before_flag = True
        if event.end < ts <= event.end + pd.Timedelta(days=event.after_days):
            after_flag = True
    return active_name, before_flag, after_flag


def _synthetic_lunar_date(ts: pd.Timestamp) -> str:
    tet_anchors = {
        2022: pd.Timestamp("2022-02-01"),
        2023: pd.Timestamp("2023-01-22"),
        2024: pd.Timestamp("2024-02-10"),
        2025: pd.Timestamp("2025-01-29"),
    }
    anchor_year = max((year for year in tet_anchors if year <= ts.year), default=min(tet_anchors))
    anchor = tet_anchors[anchor_year]
    delta_days = int((ts.normalize() - anchor).days)
    if abs(delta_days) > 45:
        return ""
    lunar_index = max(delta_days, 0)
    lunar_month = lunar_index // 29 + 1
    lunar_day = lunar_index % 29 + 1
    return f"L{lunar_month:02d}-{lunar_day:02d}"


def _build_products(config: SyntheticConfig, rng: np.random.Generator) -> pd.DataFrame:
    dry_food_names = [
        "Gạo ST25",
        "Mì Hảo Hảo Tôm Chua Cay",
        "Mì Omachi Sườn Hầm Ngũ Quả",
        "Phở Vifon bò",
        "Bún khô",
        "Miến dong",
        "Bún gạo lứt",
        "Cháo ăn liền",
        "Bột mì đa dụng",
        "Bột nêm Knorr",
        "Nước mắm truyền thống",
        "Nước tương Maggi",
        "Dầu ăn Tường An",
        "Hạt nêm Aji-ngon",
        "Sữa đặc Ông Thọ",
        "Cá hộp",
        "Thịt hộp",
        "Đậu xanh cà vỏ",
        "Đậu đỏ",
        "Yến mạch",
        "Ngũ cốc ăn sáng",
        "Mì trộn cay",
        "Bánh đa khô",
        "Bún tàu",
        "Gạo lứt hữu cơ",
    ]
    drink_names = [
        "Nước suối Lavie 500ml",
        "Nước suối Aquafina 500ml",
        "Trà xanh không độ",
        "Trà ô long",
        "Cà phê hòa tan G7",
        "Cà phê sữa lon",
        "Sữa tươi Vinamilk",
        "Sữa đậu nành",
        "Nước cam ép",
        "Nước chanh muối",
        "Nước tăng lực Number 1",
        "Nước tăng lực Red Bull",
        "Trà sữa đóng chai",
        "Sữa chua uống",
        "Nước yến",
        "Nước khoáng chai lớn",
        "Nước ép táo",
        "Nước dừa đóng chai",
        "Bia lon",
        "Nước ngọt có gas",
    ]
    snack_names = [
        "Bánh quy Cosy",
        "Bánh gạo One One",
        "Kẹo trái cây",
        "Kẹo mút",
        "Snack khoai tây",
        "Bim bim tôm",
        "Bánh xốp",
        "Socola mini",
        "Bánh trung thu mini",
        "Hạt hướng dương",
        "Đậu phộng rang",
        "Bánh que",
        "Bánh bông lan",
        "Kẹo dẻo",
        "Bánh cracker",
        "Snack rong biển",
        "Bánh trứng",
        "Kẹo sữa",
        "Bánh cookies",
        "Hạt điều rang muối",
    ]
    essential_names = [
        "Nước rửa chén",
        "Bột giặt",
        "Nước lau sàn",
        "Kem đánh răng",
        "Bàn chải đánh răng",
        "Dầu gội đầu",
        "Sữa tắm",
        "Khăn giấy",
        "Giấy vệ sinh",
        "Nước xả vải",
        "Xà phòng cục",
        "Khử mùi phòng",
        "Nước rửa tay",
        "Bông tăm",
        "Băng vệ sinh",
        "Tã giấy",
        "Dao cạo râu",
        "Lăn khử mùi",
        "Túi rác",
        "Dung dịch vệ sinh",
    ]
    household_names = [
        "Nồi inox",
        "Chảo chống dính",
        "Hộp đựng thực phẩm",
        "Bộ chén dĩa",
        "Thau nhựa",
        "Rổ nhựa",
        "Cây lau nhà",
        "Chổi quét nhà",
        "Kệ gia vị",
        "Móc treo quần áo",
        "Thùng rác gia đình",
        "Ca nhựa",
        "Bình đựng nước",
        "Khay chia ngăn",
        "Bộ dao nhà bếp",
    ]

    product_specs: list[dict[str, object]] = []
    categories = [
        ("Thực phẩm khô", dry_food_names, 25, ["Acecook", "Masan", "Vifon", "Bibica", "Tường An"]),
        ("Đồ uống", drink_names, 20, ["Vinamilk", "TH True Milk", "PepsiCo", "Coca-Cola", "Tân Hiệp Phát"]),
        ("Bánh kẹo", snack_names, 20, ["Orion", "Oishi", "Kinh Đô", "Bibica", "Hải Hà"]),
        ("Nhu yếu phẩm", essential_names, 20, ["Unilever", "P&G", "Colgate", "Diana", "SCA"]),
        ("Đồ gia dụng", household_names, 15, ["Lock&Lock", "Sunhouse", "Inochi", "Duy Tân", "Sakura"]),
    ]

    price_ranges = {
        "Thực phẩm khô": (12000, 95000),
        "Đồ uống": (7000, 28000),
        "Bánh kẹo": (5000, 45000),
        "Nhu yếu phẩm": (10000, 180000),
        "Đồ gia dụng": (20000, 320000),
    }
    size_options = {
        "Thực phẩm khô": ["200g", "500g", "1kg", "2kg", "5kg"],
        "Đồ uống": ["330ml", "500ml", "1L", "1.5L", "6x330ml"],
        "Bánh kẹo": ["50g", "100g", "150g", "200g", "500g"],
        "Nhu yếu phẩm": ["50ml", "100ml", "250ml", "500ml", "1kg"],
        "Đồ gia dụng": ["1 cái", "2 cái", "3 cái", "5 cái"],
    }
    units = {
        "Thực phẩm khô": "gói",
        "Đồ uống": "chai",
        "Bánh kẹo": "gói",
        "Nhu yếu phẩm": "chai",
        "Đồ gia dụng": "cái",
    }

    item_counter = 1
    for category_name, names, count, brands in categories:
        for idx in range(count):
            base_name = names[idx % len(names)]
            size = size_options[category_name][idx % len(size_options[category_name])]
            product_name = f"{base_name} {size}"
            if idx >= len(names):
                product_name = f"{base_name} {size} mẫu {idx - len(names) + 2}"
            base_price_low, base_price_high = price_ranges[category_name]
            base_price = int(rng.integers(base_price_low, base_price_high + 1))
            cost_price = int(round(base_price * rng.uniform(0.65, 0.82)))
            product_specs.append(
                {
                    "item_id": f"P{item_counter:03d}",
                    "product_name": product_name,
                    "category": category_name,
                    "sub_category": {
                        "Thực phẩm khô": ["Mì", "Gạo", "Gia vị", "Hạt", "Đóng hộp"],
                        "Đồ uống": ["Nước đóng chai", "Sữa", "Cà phê", "Trà", "Nước ngọt"],
                        "Bánh kẹo": ["Bánh quy", "Kẹo", "Snack", "Socola", "Hạt"],
                        "Nhu yếu phẩm": ["Chăm sóc cá nhân", "Giặt rửa", "Giấy", "Vệ sinh", "Gia dụng mềm"],
                        "Đồ gia dụng": ["Nhà bếp", "Nhà cửa", "Lưu trữ", "Dụng cụ", "Tiện ích"],
                    }[category_name][idx % 5],
                    "brand": brands[idx % len(brands)],
                    "pack_size": size,
                    "unit": units[category_name],
                    "base_price": base_price,
                    "cost_price": cost_price,
                    "product_lifecycle": ["Mới", "Tăng trưởng", "Ổn định", "Ổn định", "Cuối vòng đời"][idx % 5],
                }
            )
            item_counter += 1
            if len(product_specs) >= config.num_products:
                break
        if len(product_specs) >= config.num_products:
            break

    return pd.DataFrame(product_specs).reset_index(drop=True)


def _build_stores(config: SyntheticConfig, rng: np.random.Generator) -> pd.DataFrame:
    stores = [
        ("S001", "Tạp hóa Minh Châu", "Hà Nội", "Đống Đa", "Miền Bắc", "Tạp hóa", 130),
        ("S002", "Mini Market Phúc Lợi", "Hà Nội", "Hoàng Mai", "Miền Bắc", "Mini Market", 180),
        ("S003", "Cửa hàng tiện lợi Thành Công", "TP.HCM", "Quận 1", "Miền Nam", "Cửa hàng tiện lợi", 220),
        ("S004", "Tạp hóa Bảo An", "TP.HCM", "Bình Thạnh", "Miền Nam", "Tạp hóa", 150),
        ("S005", "Siêu thị mini Hòa Bình", "Đà Nẵng", "Hải Châu", "Miền Trung", "Siêu thị mini", 260),
        ("S006", "Cửa hàng tiện lợi Cửu Long", "Cần Thơ", "Ninh Kiều", "Miền Nam", "Cửa hàng tiện lợi", 200),
    ]
    stores = stores[: max(1, min(config.num_stores, len(stores)))]
    opening_dates = [
        "2019-08-01",
        "2020-05-15",
        "2018-11-10",
        "2019-03-20",
        "2021-01-05",
        "2020-09-09",
    ]
    rows = []
    for idx, (store_id, store_name, city, district, region, store_type, size) in enumerate(stores):
        rows.append(
            {
                "store_id": store_id,
                "store_name": store_name,
                "city": city,
                "district": district,
                "region": region,
                "store_type": store_type,
                "store_size": size,
                "opening_date": opening_dates[idx % len(opening_dates)],
            }
        )
    return pd.DataFrame(rows)


def _build_suppliers(config: SyntheticConfig, rng: np.random.Generator) -> pd.DataFrame:
    suppliers = [
        ("SUP001", "Ace Distribution", "Nhà sản xuất", "Hà Nội", 3.0, 0.96, 4.7),
        ("SUP002", "Mekong Wholesale", "Nhà phân phối", "TP.HCM", 4.0, 0.95, 4.5),
        ("SUP003", "Central Goods", "Tổng đại lý", "Đà Nẵng", 5.0, 0.94, 4.3),
        ("SUP004", "Northern FMCG", "Nhà phân phối", "Hà Nội", 3.5, 0.97, 4.8),
        ("SUP005", "Southern Supply", "Nhà sản xuất", "TP.HCM", 4.5, 0.95, 4.4),
        ("SUP006", "Retail Plus", "Tổng đại lý", "Cần Thơ", 5.5, 0.93, 4.1),
        ("SUP007", "Homeware Pro", "Nhà sản xuất", "Bình Dương", 4.0, 0.95, 4.2),
        ("SUP008", "Consumer Link", "Nhà phân phối", "Hải Phòng", 4.2, 0.94, 4.0),
    ]
    suppliers = suppliers[: max(1, min(config.num_suppliers, len(suppliers)))]
    return pd.DataFrame(
        [
            {
                "supplier_id": supplier_id,
                "supplier_name": supplier_name,
                "supplier_type": supplier_type,
                "city": city,
                "lead_time_avg": lead_time_avg,
                "fill_rate": fill_rate,
                "reliability_score": reliability_score,
            }
            for supplier_id, supplier_name, supplier_type, city, lead_time_avg, fill_rate, reliability_score in suppliers
        ]
    )


def _build_calendar(config: SyntheticConfig) -> pd.DataFrame:
    dates = pd.date_range(config.start_date, config.end_date, freq="D")
    rows = []
    for ts in dates:
        holiday_name, before_flag, after_flag = _calendar_event_for_date(ts)
        rows.append(
            {
                "date": ts.strftime("%Y-%m-%d"),
                "day_of_week": _weekday_name(ts.dayofweek),
                "week_of_year": int(ts.isocalendar().week),
                "month": int(ts.month),
                "month_name": _month_name(ts.month),
                "quarter": int(ts.quarter),
                "year": int(ts.year),
                "is_weekend": int(ts.dayofweek >= 5),
                "is_holiday": int(bool(holiday_name)),
                "holiday_name": holiday_name,
                "is_tet": int("Tết" in holiday_name),
                "is_before_holiday": int(before_flag),
                "is_after_holiday": int(after_flag),
                "lunar_date": _synthetic_lunar_date(ts),
                "season": _season_for_date(ts),
            }
        )
    return pd.DataFrame(rows)


def _build_weather(calendar: pd.DataFrame, stores: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    city_profiles = {
        "Hà Nội": {"temp": 24.0, "humid": 72.0, "rain": 3.5},
        "TP.HCM": {"temp": 28.5, "humid": 78.0, "rain": 3.0},
        "Đà Nẵng": {"temp": 26.8, "humid": 74.0, "rain": 3.8},
        "Cần Thơ": {"temp": 28.0, "humid": 80.0, "rain": 3.2},
        "Hải Phòng": {"temp": 24.5, "humid": 73.0, "rain": 3.6},
    }
    city_list = sorted(stores["city"].unique())
    rows = []
    for date_str in calendar["date"].tolist():
        ts = pd.Timestamp(date_str)
        month = ts.month
        for city in city_list:
            profile = city_profiles.get(city, {"temp": 26.0, "humid": 75.0, "rain": 3.2})
            if month in {12, 1, 2}:
                temp = profile["temp"] - 2.5
                rain_boost = 0.5
            elif month in {3, 4, 5}:
                temp = profile["temp"] + 0.2
                rain_boost = 0.8
            elif month in {6, 7, 8}:
                temp = profile["temp"] + 2.8
                rain_boost = 1.0
            else:
                temp = profile["temp"] + 0.8
                rain_boost = 1.2
            humidity = profile["humid"] + (4 if month in {6, 7, 8, 9} else -2) + rng.normal(0, 3)
            rainfall = max(0.0, profile["rain"] * rain_boost + rng.gamma(1.4, 2.0))
            if rainfall > 14:
                weather_type = "Mưa to"
            elif rainfall > 6:
                weather_type = "Mưa"
            elif temp > 30:
                weather_type = "Nắng nóng"
            elif humidity > 82:
                weather_type = "Ẩm"
            else:
                weather_type = "Nắng"
            rows.append(
                {
                    "date": date_str,
                    "city": city,
                    "temperature": round(temp + rng.normal(0, 1.2), 1),
                    "humidity": int(np.clip(round(humidity), 40, 98)),
                    "rainfall": round(rainfall, 1),
                    "weather_type": weather_type,
                }
            )
    return pd.DataFrame(rows)


def _seasonal_product_params(category: str) -> dict[str, float]:
    return {
        "Thực phẩm khô": {"base_min": 8, "base_max": 25, "variability": 0.22, "return_rate": 0.006, "order_cost": 42, "holding_cost": 1.8, "promo_sensitivity": 0.12, "holiday_sensitivity": 0.18, "weekend_sensitivity": 0.04},
        "Đồ uống": {"base_min": 10, "base_max": 35, "variability": 0.28, "return_rate": 0.004, "order_cost": 36, "holding_cost": 1.6, "promo_sensitivity": 0.18, "holiday_sensitivity": 0.26, "weekend_sensitivity": 0.10},
        "Bánh kẹo": {"base_min": 7, "base_max": 30, "variability": 0.30, "return_rate": 0.008, "order_cost": 34, "holding_cost": 1.5, "promo_sensitivity": 0.20, "holiday_sensitivity": 0.34, "weekend_sensitivity": 0.12},
        "Nhu yếu phẩm": {"base_min": 5, "base_max": 20, "variability": 0.18, "return_rate": 0.003, "order_cost": 40, "holding_cost": 1.7, "promo_sensitivity": 0.10, "holiday_sensitivity": 0.14, "weekend_sensitivity": 0.03},
        "Đồ gia dụng": {"base_min": 2, "base_max": 8, "variability": 0.24, "return_rate": 0.010, "order_cost": 55, "holding_cost": 2.1, "promo_sensitivity": 0.14, "holiday_sensitivity": 0.18, "weekend_sensitivity": 0.08},
    }[category]


def _holiday_factor(category: str, holiday_name: str, is_before_holiday: int, is_after_holiday: int, is_tet: int) -> float:
    if not holiday_name and not is_tet and not is_before_holiday and not is_after_holiday:
        return 1.0
    if category == "Thực phẩm khô":
        if is_tet:
            return 1.75
        if is_before_holiday:
            return 1.35
        if holiday_name in {"Tết Dương lịch", "Quốc khánh", "Giải phóng miền Nam", "Quốc tế Lao động"}:
            return 1.10
        return 1.05
    if category == "Đồ uống":
        if is_tet:
            return 1.35
        if holiday_name in {"Trung Thu", "Noel", "Black Friday"}:
            return 1.30
        if is_before_holiday:
            return 1.18
        return 1.08
    if category == "Bánh kẹo":
        if is_tet:
            return 1.55
        if holiday_name in {"Trung Thu", "Noel", "Black Friday"}:
            return 1.42
        if is_before_holiday:
            return 1.22
        return 1.12
    if category == "Nhu yếu phẩm":
        if is_tet:
            return 1.60
        if is_before_holiday:
            return 1.28
        return 1.06
    if category == "Đồ gia dụng":
        if holiday_name in {"Black Friday", "Noel"}:
            return 1.28
        if is_before_holiday:
            return 1.12
        return 1.03
    return 1.0


def _weekend_factor(category: str, is_weekend: int) -> float:
    if not is_weekend:
        return 1.0
    mapping = {
        "Thực phẩm khô": 1.03,
        "Đồ uống": 1.10,
        "Bánh kẹo": 1.12,
        "Nhu yếu phẩm": 0.98,
        "Đồ gia dụng": 1.05,
    }
    return mapping.get(category, 1.0)


def _season_factor(category: str, season: str) -> float:
    if category == "Đồ uống":
        return {"Hè": 1.20, "Thu": 1.05, "Xuân": 1.00, "Đông": 0.92, "Cuối năm": 1.08}.get(season, 1.0)
    if category == "Bánh kẹo":
        return {"Hè": 1.06, "Thu": 1.15, "Xuân": 1.10, "Đông": 0.95, "Cuối năm": 1.12}.get(season, 1.0)
    if category == "Thực phẩm khô":
        return {"Hè": 0.98, "Thu": 1.02, "Xuân": 1.15, "Đông": 1.05, "Cuối năm": 1.18}.get(season, 1.0)
    if category == "Nhu yếu phẩm":
        return {"Hè": 1.00, "Thu": 1.02, "Xuân": 1.08, "Đông": 0.98, "Cuối năm": 1.10}.get(season, 1.0)
    if category == "Đồ gia dụng":
        return {"Hè": 1.03, "Thu": 1.06, "Xuân": 1.10, "Đông": 1.00, "Cuối năm": 1.15}.get(season, 1.0)
    return 1.0


def _weather_factor(category: str, weather_type: str, rainfall: float, temperature: float) -> float:
    if weather_type == "Mưa to":
        return {"Thực phẩm khô": 0.98, "Đồ uống": 0.97, "Bánh kẹo": 0.95, "Nhu yếu phẩm": 1.05, "Đồ gia dụng": 0.99}.get(category, 1.0)
    if weather_type == "Mưa":
        return {"Thực phẩm khô": 0.99, "Đồ uống": 1.02, "Bánh kẹo": 0.99, "Nhu yếu phẩm": 1.04, "Đồ gia dụng": 1.00}.get(category, 1.0)
    if temperature >= 32:
        return {"Đồ uống": 1.15, "Bánh kẹo": 1.05, "Nhu yếu phẩm": 1.03, "Thực phẩm khô": 0.98, "Đồ gia dụng": 0.97}.get(category, 1.0)
    if temperature <= 20:
        return {"Đồ uống": 0.95, "Bánh kẹo": 1.02, "Nhu yếu phẩm": 1.02, "Thực phẩm khô": 1.04, "Đồ gia dụng": 1.03}.get(category, 1.0)
    return 1.0


def _select_promo_periods(
    category: str,
    store_id: str,
    item_id: str,
    calendar: pd.DataFrame,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    dates = pd.to_datetime(calendar["date"])
    years = sorted(dates.dt.year.unique())
    promo_rows: list[dict[str, object]] = []
    promo_id = 1
    for year in years:
        year_events = _holiday_events_for_year(int(year))
        relevant_events = [e for e in year_events if e.name in {"Tết Nguyên Đán", "Trung Thu", "Noel", "Black Friday", "Back to School", "Tết Dương lịch"}]
        if category in {"Thực phẩm khô", "Nhu yếu phẩm"}:
            relevant_events = [e for e in year_events if e.name in {"Tết Nguyên Đán", "Tết Dương lịch", "Quốc khánh", "Back to School"}]
        elif category == "Đồ uống":
            relevant_events = [e for e in year_events if e.name in {"Tết Nguyên Đán", "Trung Thu", "Noel", "Black Friday", "Tết Dương lịch"}]
        elif category == "Bánh kẹo":
            relevant_events = [e for e in year_events if e.name in {"Tết Nguyên Đán", "Trung Thu", "Noel", "Black Friday"}]
        elif category == "Đồ gia dụng":
            relevant_events = [e for e in year_events if e.name in {"Black Friday", "Noel", "Back to School", "Tết Nguyên Đán"}]
        if not relevant_events:
            relevant_events = year_events[:3]
        selected_events = rng.choice(relevant_events, size=min(3, len(relevant_events)), replace=False)
        for event in selected_events:
            duration = int(rng.integers(5, 14))
            calendar_start = pd.Timestamp(pd.to_datetime(calendar["date"]).min())
            calendar_end = pd.Timestamp(pd.to_datetime(calendar["date"]).max())
            start = max(event.start - pd.Timedelta(days=int(rng.integers(0, 3))), calendar_start)
            end = min(start + pd.Timedelta(days=duration), calendar_end)
            discount = int(np.clip(rng.choice([5, 10, 12, 15, 20, 25, 30]), 5, 35))
            channel = rng.choice(["Tại cửa hàng", "Facebook", "Zalo", "Ứng dụng nội bộ"])
            campaign_name = f"{event.name} {category} {year}"
            promo_rows.append(
                {
                    "promo_id": f"PR{promo_id:05d}",
                    "date_start": pd.Timestamp(start).strftime("%Y-%m-%d"),
                    "date_end": pd.Timestamp(end).strftime("%Y-%m-%d"),
                    "store_id": store_id,
                    "item_id": item_id,
                    "promo_type": rng.choice(["Giảm giá", "Combo", "Mua 1 tặng 1", "Voucher"]),
                    "discount_pct": discount,
                    "campaign_name": campaign_name,
                    "channel": channel,
                    "expected_uplift": round(0.05 + discount / 100 * rng.uniform(0.8, 1.8), 3),
                }
            )
            promo_id += 1
    return promo_rows


def _promo_lookup(promo_rows: pd.DataFrame) -> dict[tuple[str, str], list[dict[str, object]]]:
    lookup: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in promo_rows.to_dict("records"):
        lookup[(row["store_id"], row["item_id"])].append(row)
    for key in lookup:
        lookup[key] = sorted(lookup[key], key=lambda r: r["date_start"])
    return lookup


def _active_promo(promo_schedule: list[dict[str, object]], ts: pd.Timestamp) -> tuple[int, int, str]:
    for promo in promo_schedule:
        start = pd.Timestamp(promo["date_start"])
        end = pd.Timestamp(promo["date_end"])
        if start <= ts <= end:
            return 1, int(promo["discount_pct"]), str(promo["campaign_name"])
    return 0, 0, ""


def _category_base_demand(category: str, rng: np.random.Generator) -> float:
    params = _seasonal_product_params(category)
    return float(rng.integers(params["base_min"], params["base_max"] + 1))


def _supplier_for_category(category: str, suppliers: pd.DataFrame) -> str:
    preferred = {
        "Thực phẩm khô": ["SUP001", "SUP004"],
        "Đồ uống": ["SUP002", "SUP005"],
        "Bánh kẹo": ["SUP001", "SUP002", "SUP004"],
        "Nhu yếu phẩm": ["SUP003", "SUP006"],
        "Đồ gia dụng": ["SUP007", "SUP008"],
    }
    options = [sid for sid in preferred.get(category, suppliers["supplier_id"].tolist()) if sid in set(suppliers["supplier_id"])]
    return options[0] if options else suppliers["supplier_id"].iloc[0]


def _simulate_pair(
    store: pd.Series,
    product: pd.Series,
    supplier: pd.Series,
    calendar: pd.DataFrame,
    weather: pd.DataFrame,
    promo_schedule: list[dict[str, object]],
    rng: np.random.Generator,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    category = str(product["category"])
    store_factor = 0.92 + (float(store["store_size"]) / 300.0)
    category_params = _seasonal_product_params(category)
    base_daily = _category_base_demand(category, rng) * store_factor
    lead_time_days = int(np.clip(round(float(supplier["lead_time_avg"]) + rng.normal(0, 1)), 2, 10))
    variability = float(category_params["variability"])
    demand_std = max(base_daily * variability, 1.0)
    safety_stock = float(1.65 * demand_std * np.sqrt(max(lead_time_days, 1)))
    reorder_point = float(base_daily * lead_time_days + safety_stock)
    annual_demand = base_daily * 365.0
    eoq = float(np.sqrt((2 * annual_demand * float(category_params["order_cost"])) / float(category_params["holding_cost"])))

    promo_lookup_rows = promo_schedule
    incoming: defaultdict[pd.Timestamp, int] = defaultdict(int)
    next_order_allowed = start_date
    inventory_level = int(max(round(reorder_point + eoq * 0.75), round(base_daily * 14)))

    sales_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    po_rows: list[dict[str, object]] = []
    return_rows: list[dict[str, object]] = []

    dates = pd.to_datetime(calendar["date"])
    calendar_index = {pd.Timestamp(row["date"]): row for row in calendar.to_dict("records")}
    weather_lookup = {
        (pd.Timestamp(row["date"]), row["city"]): row for row in weather.to_dict("records")
    }
    city = str(store["city"])
    product_name = str(product["product_name"])
    product_category = str(product["category"])

    for ts in dates:
        cal = calendar_index[ts]
        weather_row = weather_lookup.get((ts, city), {})
        received_qty = int(incoming.pop(ts, 0))
        stock_begin = int(inventory_level)
        available = stock_begin + received_qty

        active_flag, discount_pct, campaign_name = _active_promo(promo_lookup_rows, ts)
        holiday_name = str(cal["holiday_name"])
        holiday_factor = _holiday_factor(product_category, holiday_name, int(cal["is_before_holiday"]), int(cal["is_after_holiday"]), int(cal["is_tet"]))
        weekend_factor = _weekend_factor(product_category, int(cal["is_weekend"]))
        season_factor = _season_factor(product_category, str(cal["season"]))
        weather_factor = _weather_factor(product_category, str(weather_row.get("weather_type", "")), float(weather_row.get("rainfall", 0.0)), float(weather_row.get("temperature", 25.0)))
        promo_factor = 1.0 + (discount_pct / 100.0) * float(category_params["promo_sensitivity"]) if active_flag else 1.0
        if campaign_name and "Tết" in campaign_name:
            promo_factor += 0.08
        trend_factor = 1.0 + (ts.year - start_date.year) * 0.025 + (ts.dayofyear / 365.0) * 0.01
        weekday_profile = [0.96, 0.98, 1.00, 1.00, 1.04, 1.10, 1.08][ts.dayofweek]
        noise_factor = float(np.clip(rng.normal(1.0, 0.10), 0.75, 1.35))
        demand_mu = max(
            0.2,
            base_daily
            * trend_factor
            * weekday_profile
            * weekend_factor
            * season_factor
            * holiday_factor
            * weather_factor
            * promo_factor
            * noise_factor,
        )
        real_demand = int(rng.poisson(lam=demand_mu))
        sold_qty = int(min(real_demand, available))
        lost_sales = int(max(real_demand - sold_qty, 0))
        inventory_level = int(max(available - sold_qty, 0))
        stock_end = int(inventory_level)
        stockout_flag = int(lost_sales > 0)

        if inventory_level <= reorder_point and ts >= next_order_allowed:
            order_qty = int(max(round(eoq), round(base_daily * (lead_time_days + 7))))
            order_qty = int(max(10, round(order_qty / 10) * 10))
            expected_delivery = ts + pd.Timedelta(days=lead_time_days)
            delay_days = int(max(0, min(4, round(rng.normal(1.0, 1.2)))))
            actual_delivery = expected_delivery + pd.Timedelta(days=delay_days)
            incoming[pd.Timestamp(actual_delivery)] += order_qty
            next_order_allowed = actual_delivery + pd.Timedelta(days=3)
            po_rows.append(
                {
                    "po_id": f"PO{len(po_rows)+1:06d}",
                    "order_date": ts.strftime("%Y-%m-%d"),
                    "expected_delivery_date": expected_delivery.strftime("%Y-%m-%d"),
                    "actual_delivery_date": actual_delivery.strftime("%Y-%m-%d"),
                    "supplier_id": str(supplier["supplier_id"]),
                    "store_id": str(store["store_id"]),
                    "item_id": str(product["item_id"]),
                    "ordered_qty": order_qty,
                    "received_qty": order_qty,
                    "delay_days": delay_days,
                    "order_status": "Đúng hạn" if delay_days == 0 else "Trễ",
                    "product_name": product_name,
                    "category": product_category,
                    "store_name": str(store["store_name"]),
                    "city": city,
                    "district": str(store["district"]),
                }
            )

        return_rate = 0.0
        if sold_qty > 0:
            return_rate = min(0.05, float(category_params["return_rate"]) * (1.0 + 0.3 * active_flag))
        returned_qty = int(rng.binomial(sold_qty, return_rate)) if sold_qty > 0 and return_rate > 0 else 0
        if returned_qty > 0:
            return_rows.append(
                {
                    "date": ts.strftime("%Y-%m-%d"),
                    "store_id": str(store["store_id"]),
                    "item_id": str(product["item_id"]),
                    "returned_qty": returned_qty,
                    "return_reason": rng.choice(["Đổi ý", "Lỗi bao bì", "Hàng cận date", "Giao sai mẫu"]),
                    "refund_amount": int(round(returned_qty * float(product["base_price"]) * (1 - discount_pct / 100.0))),
                    "product_name": product_name,
                    "category": product_category,
                    "store_name": str(store["store_name"]),
                    "city": city,
                    "district": str(store["district"]),
                }
            )

        unit_price = int(round(float(product["base_price"]) * (1.0 + 0.025 * (ts.year - start_date.year)) * (1.0 - discount_pct / 100.0)))
        unit_price = max(unit_price, int(product["cost_price"]))
        revenue = int(sold_qty * unit_price)

        sales_rows.append(
            {
                "date": ts.strftime("%Y-%m-%d"),
                "store_id": str(store["store_id"]),
                "item_id": str(product["item_id"]),
                "quantity_sold": sold_qty,
                "unit_price": unit_price,
                "revenue": revenue,
                "discount_pct": discount_pct,
                "promo_flag": active_flag,
                "stock_begin": stock_begin,
                "stock_end": stock_end,
                "stockout_flag": stockout_flag,
                "day_of_week": cal["day_of_week"],
                "week_of_year": int(cal["week_of_year"]),
                "month": int(cal["month"]),
                "quarter": int(cal["quarter"]),
                "year": int(cal["year"]),
                "holiday_flag": int(cal["is_holiday"]),
                "holiday_name": holiday_name,
                "is_tet": int(cal["is_tet"]),
                "season": cal["season"],
                "product_name": product_name,
                "category": product_category,
                "brand": str(product["brand"]),
                "pack_size": str(product["pack_size"]),
                "unit": str(product["unit"]),
                "base_price": int(product["base_price"]),
                "cost_price": int(product["cost_price"]),
                "product_lifecycle": str(product["product_lifecycle"]),
                "store_name": str(store["store_name"]),
                "city": city,
                "district": str(store["district"]),
                "region": str(store["region"]),
                "store_type": str(store["store_type"]),
                "store_size": int(store["store_size"]),
            }
        )

        inventory_rows.append(
            {
                "date": ts.strftime("%Y-%m-%d"),
                "store_id": str(store["store_id"]),
                "item_id": str(product["item_id"]),
                "inventory_begin": stock_begin,
                "inventory_end": stock_end,
                "received_qty": received_qty,
                "sold_qty": sold_qty,
                "lost_sales": lost_sales,
                "stockout_flag": stockout_flag,
                "safety_stock": int(round(safety_stock)),
                "reorder_point": int(round(reorder_point)),
                "eoq": int(round(eoq)),
                "lead_time_days": lead_time_days,
                "product_name": product_name,
                "category": product_category,
                "brand": str(product["brand"]),
                "pack_size": str(product["pack_size"]),
                "unit": str(product["unit"]),
                "base_price": int(product["base_price"]),
                "cost_price": int(product["cost_price"]),
                "store_name": str(store["store_name"]),
                "city": city,
                "district": str(store["district"]),
                "region": str(store["region"]),
                "store_type": str(store["store_type"]),
                "store_size": int(store["store_size"]),
            }
        )

    return pd.DataFrame(sales_rows), pd.DataFrame(inventory_rows), po_rows, return_rows


def _write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def generate_vn_retail_dataset(
    output_dir: str | Path,
    start_date: str | pd.Timestamp = "2022-01-01",
    end_date: str | pd.Timestamp = "2024-12-31",
    num_products: int = 100,
    num_stores: int = 5,
    num_suppliers: int = 8,
    seed: int = 42,
) -> dict[str, Path]:
    config = SyntheticConfig(
        output_dir=Path(output_dir),
        start_date=pd.Timestamp(start_date),
        end_date=pd.Timestamp(end_date),
        num_products=num_products,
        num_stores=num_stores,
        num_suppliers=num_suppliers,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob("*.csv"):
        existing.unlink()

    products = _build_products(config, rng)
    stores = _build_stores(config, rng)
    suppliers = _build_suppliers(config, rng)
    calendar = _build_calendar(config)
    weather = _build_weather(calendar, stores, rng)

    promo_rows: list[dict[str, object]] = []
    sales_path = output_dir / "fact_sales.csv"
    inventory_path = output_dir / "fact_inventory.csv"
    purchase_orders_path = output_dir / "fact_purchase_orders.csv"
    returns_path = output_dir / "fact_returns.csv"

    if sales_path.exists():
        sales_path.unlink()
    if inventory_path.exists():
        inventory_path.unlink()
    if purchase_orders_path.exists():
        purchase_orders_path.unlink()
    if returns_path.exists():
        returns_path.unlink()

    sales_frames: list[pd.DataFrame] = []
    inventory_frames: list[pd.DataFrame] = []
    po_rows_all: list[dict[str, object]] = []
    return_rows_all: list[dict[str, object]] = []

    for store in stores.itertuples(index=False):
        for product in products.itertuples(index=False):
            supplier_id = _supplier_for_category(str(product.category), suppliers)
            supplier = suppliers[suppliers["supplier_id"] == supplier_id].iloc[0]
            pair_promos = _select_promo_periods(str(product.category), str(store.store_id), str(product.item_id), calendar, rng)
            promo_rows.extend(pair_promos)
            sales_df, inventory_df, po_rows, return_rows = _simulate_pair(
                pd.Series(store._asdict()),
                pd.Series(product._asdict()),
                supplier,
                calendar,
                weather,
                pair_promos,
                rng,
                config.start_date,
                config.end_date,
            )
            sales_frames.append(sales_df)
            inventory_frames.append(inventory_df)
            po_rows_all.extend(po_rows)
            return_rows_all.extend(return_rows)

            if len(sales_frames) >= 5:
                chunk_sales = pd.concat(sales_frames, ignore_index=True)
                _append_csv(chunk_sales, sales_path)
                sales_frames = []
            if len(inventory_frames) >= 5:
                chunk_inventory = pd.concat(inventory_frames, ignore_index=True)
                _append_csv(chunk_inventory, inventory_path)
                inventory_frames = []

    if sales_frames:
        _append_csv(pd.concat(sales_frames, ignore_index=True), sales_path)
    if inventory_frames:
        _append_csv(pd.concat(inventory_frames, ignore_index=True), inventory_path)

    dim_product_path = output_dir / "dim_product.csv"
    dim_store_path = output_dir / "dim_store.csv"
    dim_calendar_path = output_dir / "dim_calendar.csv"
    dim_supplier_path = output_dir / "dim_supplier.csv"
    fact_promotion_path = output_dir / "fact_promotion.csv"
    fact_weather_path = output_dir / "fact_weather.csv"

    _write_df(products, dim_product_path)
    _write_df(stores, dim_store_path)
    _write_df(calendar, dim_calendar_path)
    _write_df(suppliers, dim_supplier_path)
    _write_df(pd.DataFrame(promo_rows).sort_values(["date_start", "store_id", "item_id"]), fact_promotion_path)
    _write_df(weather, fact_weather_path)
    _write_df(pd.DataFrame(po_rows_all).sort_values(["order_date", "store_id", "item_id"]), purchase_orders_path)
    _write_df(pd.DataFrame(return_rows_all).sort_values(["date", "store_id", "item_id"]), returns_path)

    fact_history_path = output_dir / "fact_history.csv"
    history_df = pd.read_csv(sales_path, low_memory=False).rename(
        columns={
            "quantity_sold": "target",
            "unit_price": "price",
            "promo_flag": "promo",
            "holiday_flag": "holiday",
        }
    )
    _write_df(history_df, fact_history_path)

    inventory_recommendations_path = output_dir / "inventory_recommendations.csv"
    inventory_df = pd.read_csv(inventory_path, low_memory=False).sort_values(["store_id", "item_id", "date"])
    recommendations = (
        inventory_df.groupby(["store_id", "item_id"], as_index=False)
        .agg(
            avg_demand=("sold_qty", "mean"),
            demand_std=("sold_qty", "std"),
            safety_stock=("safety_stock", "max"),
            reorder_point=("reorder_point", "max"),
            eoq=("eoq", "max"),
            lead_time_days=("lead_time_days", "max"),
            inventory_begin=("inventory_begin", "first"),
            inventory_end=("inventory_end", "last"),
            product_name=("product_name", "last"),
            category=("category", "last"),
            brand=("brand", "last"),
            pack_size=("pack_size", "last"),
            unit=("unit", "last"),
            base_price=("base_price", "last"),
            cost_price=("cost_price", "last"),
            store_name=("store_name", "last"),
            city=("city", "last"),
            district=("district", "last"),
            region=("region", "last"),
            store_type=("store_type", "last"),
            store_size=("store_size", "last"),
        )
        .fillna(0)
    )
    recommendations["source_model"] = "synthetic-rule"
    _write_df(recommendations, inventory_recommendations_path)

    return {
        "fact_sales": sales_path,
        "fact_history": fact_history_path,
        "dim_product": dim_product_path,
        "dim_store": dim_store_path,
        "dim_calendar": dim_calendar_path,
        "fact_inventory": inventory_path,
        "inventory_recommendations": inventory_recommendations_path,
        "fact_promotion": fact_promotion_path,
        "fact_purchase_orders": purchase_orders_path,
        "dim_supplier": dim_supplier_path,
        "fact_returns": returns_path,
        "fact_weather": fact_weather_path,
    }


def _append_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    header = not path.exists()
    frame.to_csv(path, mode="a", header=header, index=False, encoding="utf-8")
