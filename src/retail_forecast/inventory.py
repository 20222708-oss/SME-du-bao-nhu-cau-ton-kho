from __future__ import annotations

import math


def z_value(service_level: float) -> float:
    """Approximate z-score using common service levels."""
    lookup = {
        0.80: 0.84,
        0.85: 1.04,
        0.90: 1.28,
        0.95: 1.65,
        0.97: 1.88,
        0.98: 2.05,
        0.99: 2.33,
    }
    rounded = round(service_level, 2)
    if rounded in lookup:
        return lookup[rounded]
    return 1.65


def safety_stock(demand_std: float, lead_time_periods: float, service_level: float = 0.95) -> float:
    return z_value(service_level) * demand_std * math.sqrt(max(lead_time_periods, 0.0))


def reorder_point(avg_demand: float, lead_time_periods: float, demand_std: float, service_level: float = 0.95) -> float:
    return avg_demand * lead_time_periods + safety_stock(demand_std, lead_time_periods, service_level)


def eoq(annual_demand: float, order_cost: float, holding_cost: float) -> float:
    if order_cost <= 0 or holding_cost <= 0:
        return 0.0
    return math.sqrt((2 * annual_demand * order_cost) / holding_cost)


def inventory_recommendation(
    forecast,
    lead_time_periods: float = 7,
    service_level: float = 0.95,
    order_cost: float = 50.0,
    holding_cost: float = 2.0,
):
    forecast_values = list(map(float, forecast))
    if not forecast_values:
        return {
            "avg_demand": 0.0,
            "demand_std": 0.0,
            "safety_stock": 0.0,
            "reorder_point": 0.0,
            "eoq": 0.0,
        }

    avg_demand = sum(forecast_values) / len(forecast_values)
    variance = sum((x - avg_demand) ** 2 for x in forecast_values) / max(len(forecast_values) - 1, 1)
    demand_std = math.sqrt(variance)
    annual_demand = avg_demand * 365.0

    return {
        "avg_demand": avg_demand,
        "demand_std": demand_std,
        "safety_stock": safety_stock(demand_std, lead_time_periods, service_level),
        "reorder_point": reorder_point(avg_demand, lead_time_periods, demand_std, service_level),
        "eoq": eoq(annual_demand, order_cost, holding_cost),
    }
