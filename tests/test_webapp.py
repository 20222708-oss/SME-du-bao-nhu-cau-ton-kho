from __future__ import annotations

from pathlib import Path

import pytest

from retail_forecast.webapp import DashboardRepository


def test_dashboard_repository_bootstrap():
    data_root = Path("synthetic_data/vn_retail_100sku_3y")
    if not data_root.exists():
        pytest.skip("Synthetic retail dataset is not available in this workspace.")

    repo = DashboardRepository(data_root)
    bootstrap = repo.bootstrap(
        store_id=None,
        item_id=None,
        category=None,
        model_name="ensemble",
        horizon=30,
        history_window=120,
        inventory_limit=8,
    )

    assert "options" in bootstrap
    assert "dashboard" in bootstrap
    assert bootstrap["dashboard"]["summary"]["product_count"] >= 1
    assert bootstrap["dashboard"]["summary"]["store_count"] >= 1
    assert isinstance(bootstrap["dashboard"]["series"]["actual"], list)
