const state = {
  store_id: "all",
  item_id: "all",
  category: "all",
  model_name: "ensemble",
  horizon: 30,
  history_window: 120,
  inventory_limit: 12,
};

const els = {};

function $(selector) {
  return document.querySelector(selector);
}

function setLoading(on) {
  const overlay = $("#loadingOverlay");
  if (!overlay) return;
  overlay.classList.toggle("hidden", !on);
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "0";
  }
  return new Intl.NumberFormat("vi-VN").format(Number(value));
}

function formatCompact(value) {
  const num = Number(value || 0);
  const abs = Math.abs(num);
  if (abs >= 1_000_000_000) return `${(num / 1_000_000_000).toFixed(1).replace(".", ",")} tỷ`;
  if (abs >= 1_000_000) return `${(num / 1_000_000).toFixed(1).replace(".", ",")} tr`;
  if (abs >= 1_000) return `${(num / 1_000).toFixed(1).replace(".", ",")} nghìn`;
  return new Intl.NumberFormat("vi-VN").format(num);
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "0%";
  return `${Number(value).toFixed(1).replace(".", ",")}%`;
}

function formatDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("vi-VN", { day: "2-digit", month: "2-digit", year: "numeric" }).format(date);
}

function toQuery(params) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      search.set(key, value);
    }
  });
  return search.toString();
}

async function fetchJson(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }
  return response.json();
}

function populateSelect(select, options, value) {
  select.innerHTML = "";
  options.forEach((option) => {
    const el = document.createElement("option");
    el.value = option.value;
    el.textContent = option.label;
    select.appendChild(el);
  });
  select.value = value;
}

function renderKpis(summary) {
  const cards = [
    {
      label: "Tổng thực tế",
      value: formatCompact(summary.total_units),
      note: "Đơn vị bán ra",
    },
    {
      label: "Dự báo 30 ngày",
      value: formatCompact(summary.forecast_units),
      note: "Forecast đang chọn",
    },
    {
      label: "SKU theo dõi",
      value: formatNumber(summary.product_count),
      note: "Sản phẩm trong phạm vi",
    },
    {
      label: "Cửa hàng",
      value: formatNumber(summary.store_count),
      note: "Điểm bán đang theo dõi",
    },
    {
      label: "Tỷ lệ stockout",
      value: formatPercent(summary.stockout_rate),
      note: "Tín hiệu hết hàng",
    },
    {
      label: "Safety stock TB",
      value: formatCompact(summary.safety_stock_avg),
      note: "Mức tồn kho an toàn",
    },
    {
      label: "Reorder point TB",
      value: formatCompact(summary.reorder_point_avg),
      note: "Điểm đặt hàng lại",
    },
    {
      label: "Độ phủ mô hình",
      value: formatNumber(summary.coverage),
      note: "Số nhóm được dự báo",
    },
  ];
  els.kpiGrid.innerHTML = cards
    .map(
      (card) => `
      <article class="panel kpi-card">
        <div class="kpi-label">${card.label}</div>
        <div class="kpi-value">${card.value}</div>
        <div class="kpi-note">${card.note}</div>
      </article>`
    )
    .join("");
}

function renderMetrics(metrics) {
  if (!metrics || !metrics.length) {
    els.metricsList.innerHTML = `<div class="metric-row"><div class="metric-name">Chưa có model_metrics.csv</div><div class="metric-muted">-</div><div class="metric-muted">-</div><div class="metric-muted">-</div></div>`;
    return;
  }
  els.metricsList.innerHTML = metrics
    .map(
      (row) => `
      <div class="metric-row">
        <div class="metric-name">${row.model_name ?? "-"}</div>
        <div class="metric-muted">Rows: ${formatNumber(row.rows ?? 0)}</div>
        <div class="metric-muted">Groups: ${formatNumber(row.groups ?? 0)}</div>
        <div class="metric-muted">${formatDate(row.min_forecast_date ?? "")} → ${formatDate(row.max_forecast_date ?? "")}</div>
      </div>`
    )
    .join("");
}

function renderSummary(summary) {
  const rows = [
    { icon: "∑", title: "Tổng doanh số", value: formatCompact(summary.total_units), note: "Tổng số lượng bán ra trong phạm vi lọc" },
    { icon: "ƒ", title: "Forecast 30 ngày", value: formatCompact(summary.forecast_units), note: "Tổng nhu cầu dự báo cho horizon hiện tại" },
    { icon: "#", title: "Danh mục", value: formatNumber(summary.category_count), note: "Số nhóm hàng đang xuất hiện" },
    { icon: "⚠", title: "Độ phủ dự báo", value: formatNumber(summary.coverage), note: "Số nhóm store-item đã được dự báo" },
    { icon: "⌁", title: "Stockout", value: formatPercent(summary.stockout_rate), note: "Tỷ lệ dòng có tín hiệu hết hàng" },
    { icon: "✓", title: "Tồn kho an toàn TB", value: formatCompact(summary.safety_stock_avg), note: "Trung bình trên inventory recommendations" },
  ];
  els.summaryStack.innerHTML = rows
    .map(
      (row) => `
      <div class="summary-row">
        <div class="summary-icon">${row.icon}</div>
        <div>
          <div class="summary-title">${row.title}</div>
          <div class="summary-value">${row.value}</div>
          <div class="summary-note">${row.note}</div>
        </div>
      </div>`
    )
    .join("");
}

function renderMetricStrip(summary) {
  const items = [
    ["Revenue", formatCompact(summary.total_revenue)],
    ["Forecast rate", formatCompact(summary.forecast_rate)],
    ["EOQ TB", formatCompact(summary.eoq_avg)],
    ["Last update", summary.actual_end ? formatDate(summary.actual_end) : "-"],
  ];
  els.metricStrip.innerHTML = items
    .map(
      ([label, value]) => `
      <div class="metric-pill">
        <div class="metric-pill-label">${label}</div>
        <div class="metric-pill-value">${value}</div>
      </div>`
    )
    .join("");
}

function renderCategoryBars(categories) {
  if (!categories || !categories.length) {
    els.categoryBars.innerHTML = `<div class="metric-muted">Không có đủ dữ liệu để phân nhóm danh mục.</div>`;
    return;
  }
  const max = Math.max(...categories.map((item) => Number(item.value || 0)), 1);
  els.categoryBars.innerHTML = categories
    .map((item) => {
      const width = (Number(item.value || 0) / max) * 100;
      return `
        <div class="bar-item">
          <div class="bar-label">${item.label}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:${Math.max(10, width)}%"></div>
          </div>
          <div class="bar-value">${formatCompact(item.value)}</div>
        </div>`;
    })
    .join("");
}

function renderInventoryTable(rows) {
  if (!rows || !rows.length) {
    els.inventoryBody.innerHTML = `<tr><td colspan="7" class="metric-muted">Chưa có inventory_recommendations.csv hoặc không có dữ liệu khớp bộ lọc.</td></tr>`;
    return;
  }
  els.inventoryBody.innerHTML = rows
    .map(
      (row) => `
      <tr>
        <td>${row.store_name ?? row.store_id ?? "-"}</td>
        <td>${row.product_name ?? row.item_id ?? "-"}</td>
        <td>${row.category ?? "-"}</td>
        <td>${formatCompact(row.avg_demand)}</td>
        <td>${formatCompact(row.safety_stock)}</td>
        <td>${formatCompact(row.reorder_point)}</td>
        <td>${formatCompact(row.eoq)}</td>
      </tr>`
    )
    .join("");
}

function renderTopProducts(rows) {
  if (!rows || !rows.length) {
    els.topProductsBody.innerHTML = `<tr><td colspan="4" class="metric-muted">Không có dữ liệu top sản phẩm trong phạm vi lọc.</td></tr>`;
    return;
  }
  els.topProductsBody.innerHTML = rows
    .map(
      (row) => `
      <tr>
        <td>${row.product_name ?? row.item_id ?? "-"}</td>
        <td>${row.category ?? "-"}</td>
        <td>${formatCompact(row.actual_units)}</td>
        <td>${formatCompact(row.forecast_units ?? 0)}</td>
      </tr>`
    )
    .join("");
}

function renderTrendChart(series) {
  const svg = els.trendChart;
  const actual = (series?.actual || []).map((item) => ({ date: item.date, value: Number(item.value ?? 0) }));
  const forecast = (series?.forecast || []).map((item) => ({ date: item.date, value: Number(item.value ?? 0) }));
  const points = [...actual, ...forecast];

  if (!points.length) {
    svg.innerHTML = `<foreignObject x="0" y="0" width="100%" height="100%"><div class="chart-empty">Không có chuỗi thời gian cho bộ lọc hiện tại.</div></foreignObject>`;
    return;
  }

  const width = 1000;
  const height = 420;
  const pad = { top: 28, right: 32, bottom: 52, left: 68 };
  const dates = points.map((item) => new Date(item.date).getTime()).filter((value) => !Number.isNaN(value));
  const minX = Math.min(...dates);
  const maxX = Math.max(...dates);
  const maxY = Math.max(...points.map((item) => item.value), 1);
  const minY = 0;
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;

  const xScale = (date) => {
    const stamp = new Date(date).getTime();
    if (maxX === minX) return pad.left + innerW / 2;
    return pad.left + ((stamp - minX) / (maxX - minX)) * innerW;
  };

  const yScale = (value) => pad.top + innerH - ((value - minY) / (maxY - minY || 1)) * innerH;

  const toPath = (dataset) => {
    if (!dataset.length) return "";
    return dataset
      .map((item, index) => {
        const x = xScale(item.date);
        const y = yScale(item.value);
        return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
      })
      .join(" ");
  };

  const areaPath = () => {
    if (!actual.length) return "";
    const startX = xScale(actual[0].date);
    const endX = xScale(actual[actual.length - 1].date);
    const baseY = yScale(0);
    return `${toPath(actual)} L ${endX.toFixed(2)} ${baseY.toFixed(2)} L ${startX.toFixed(2)} ${baseY.toFixed(2)} Z`;
  };

  const gridLines = [];
  for (let i = 0; i <= 4; i += 1) {
    const y = pad.top + (innerH / 4) * i;
    const value = Math.round(maxY - (maxY / 4) * i);
    gridLines.push(`
      <line x1="${pad.left}" y1="${y}" x2="${width - pad.right}" y2="${y}" class="grid-line"></line>
      <text x="${pad.left - 12}" y="${y + 4}" class="axis-label axis-label-y" text-anchor="end">${formatCompact(value)}</text>
    `);
  }

  const labelSource = actual.length ? actual : forecast;
  const labelIndices = [0, Math.round(labelSource.length / 2), labelSource.length - 1].filter(
    (idx, pos, arr) => idx >= 0 && arr.indexOf(idx) === pos
  );
  const xLabels = labelIndices
    .map((index) => {
      const item = labelSource[index];
      if (!item) return "";
      const x = xScale(item.date);
      return `<text x="${x}" y="${height - 16}" class="axis-label" text-anchor="middle">${formatDate(item.date)}</text>`;
    })
    .join("");

  const lastActual = actual[actual.length - 1];
  const lastForecast = forecast[forecast.length - 1];

  svg.innerHTML = `
    <defs>
      <linearGradient id="fillArea" x1="0" x2="0" y1="0" y2="1">
        <stop offset="0%" stop-color="rgba(126,168,255,0.42)"></stop>
        <stop offset="100%" stop-color="rgba(126,168,255,0.03)"></stop>
      </linearGradient>
    </defs>
    ${gridLines.join("")}
    <path d="${areaPath()}" class="area-path"></path>
    <path d="${toPath(actual)}" class="trend-path trend-actual"></path>
    <path d="${toPath(forecast)}" class="trend-path trend-forecast"></path>
    ${
      lastActual
        ? `<circle cx="${xScale(lastActual.date)}" cy="${yScale(lastActual.value)}" r="5" class="point point-actual"></circle>`
        : ""
    }
    ${
      lastForecast
        ? `<circle cx="${xScale(lastForecast.date)}" cy="${yScale(lastForecast.value)}" r="5" class="point point-forecast"></circle>`
        : ""
    }
    ${xLabels}
  `;
}

function updateStatus(payload) {
  els.lastRefresh.textContent = new Intl.DateTimeFormat("vi-VN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  }).format(new Date());
  els.coverageValue.textContent = formatNumber(payload.summary?.coverage ?? 0);
  els.dataRootLabel.textContent = payload.data_root || "-";
  els.artifactRootLabel.textContent = payload.artifacts_root ? `Artifacts: ${payload.artifacts_root}` : "Artifacts: lấy trực tiếp từ dữ liệu gốc";
}

function syncStateFromUI() {
  state.store_id = els.storeFilter.value;
  state.item_id = els.productFilter.value;
  state.category = els.categoryFilter.value;
  state.model_name = els.modelFilter.value;
  state.horizon = Number(els.horizonInput.value || 30);
  state.history_window = Number(els.windowInput.value || 120);
}

async function loadDashboard() {
  setLoading(true);
  syncStateFromUI();
  const query = toQuery(state);
  const payload = await fetchJson(`/api/dashboard?${query}`);
  currentDashboard = payload;
  renderKpis(payload.summary);
  renderSummary(payload.summary);
  renderMetricStrip(payload.summary);
  renderCategoryBars(payload.categories);
  renderTrendChart(payload.series);
  renderInventoryTable(payload.inventory);
  renderTopProducts(payload.top_products);
  renderMetrics(payload.metrics);
  updateStatus({
    ...payload,
    data_root: bootstrapInfo?.data_root,
    artifacts_root: bootstrapInfo?.artifacts_root,
  });
  setLoading(false);
}

let bootstrapInfo = null;
let currentDashboard = null;

function populateFilters(options) {
  populateSelect(els.storeFilter, [{ value: "all", label: "Tất cả cửa hàng" }, ...options.stores], state.store_id);
  populateSelect(els.productFilter, [{ value: "all", label: "Tất cả sản phẩm" }, ...options.products], state.item_id);
  populateSelect(els.categoryFilter, [{ value: "all", label: "Tất cả danh mục" }, ...options.categories], state.category);
  populateSelect(els.modelFilter, options.models, state.model_name);
}

function wireEvents() {
  [els.storeFilter, els.productFilter, els.categoryFilter, els.modelFilter, els.horizonInput, els.windowInput].forEach(
    (element) => {
      element.addEventListener("change", () => loadDashboard().catch(showError));
    }
  );
  els.refreshBtn.addEventListener("click", () => loadDashboard().catch(showError));
  els.resetBtn.addEventListener("click", async () => {
    state.store_id = "all";
    state.item_id = "all";
    state.category = "all";
    state.model_name = "ensemble";
    state.horizon = 30;
    state.history_window = 120;
    els.horizonInput.value = "30";
    els.windowInput.value = "120";
    populateFilters(bootstrapInfo.options);
    await loadDashboard().catch(showError);
  });
  window.addEventListener("resize", () => {
    if (currentDashboard) {
      renderTrendChart(currentDashboard.series);
    }
  });
}

function showError(error) {
  console.error(error);
  setLoading(false);
  els.summaryStack.innerHTML = `
    <div class="summary-row">
      <div class="summary-icon">!</div>
      <div>
        <div class="summary-title">Không tải được dữ liệu</div>
        <div class="summary-value">Kiểm tra lại API hoặc đường dẫn dữ liệu</div>
        <div class="summary-note">${String(error.message || error)}</div>
      </div>
    </div>`;
}

async function init() {
  els.kpiGrid = $("#kpiGrid");
  els.trendChart = $("#trendChart");
  els.categoryBars = $("#categoryBars");
  els.metricsList = $("#metricsList");
  els.summaryStack = $("#summaryStack");
  els.metricStrip = $("#metricStrip");
  els.inventoryBody = $("#inventoryBody");
  els.topProductsBody = $("#topProductsBody");
  els.lastRefresh = $("#lastRefresh");
  els.coverageValue = $("#coverageValue");
  els.dataRootLabel = $("#dataRootLabel");
  els.artifactRootLabel = $("#artifactRootLabel");
  els.storeFilter = $("#storeFilter");
  els.productFilter = $("#productFilter");
  els.categoryFilter = $("#categoryFilter");
  els.modelFilter = $("#modelFilter");
  els.horizonInput = $("#horizonInput");
  els.windowInput = $("#windowInput");
  els.refreshBtn = $("#refreshBtn");
  els.resetBtn = $("#resetBtn");

  wireEvents();

  setLoading(true);
  bootstrapInfo = await fetchJson(`/api/bootstrap?${toQuery(state)}`);
  populateFilters(bootstrapInfo.options);
  renderKpis(bootstrapInfo.dashboard.summary);
  renderSummary(bootstrapInfo.dashboard.summary);
  renderMetricStrip(bootstrapInfo.dashboard.summary);
  renderCategoryBars(bootstrapInfo.dashboard.categories);
  renderTrendChart(bootstrapInfo.dashboard.series);
  renderInventoryTable(bootstrapInfo.dashboard.inventory);
  renderTopProducts(bootstrapInfo.dashboard.top_products);
  renderMetrics(bootstrapInfo.dashboard.metrics);
  updateStatus({
    ...bootstrapInfo.dashboard,
    data_root: bootstrapInfo.data_root,
    artifacts_root: bootstrapInfo.artifacts_root,
  });
  currentDashboard = bootstrapInfo.dashboard;
  setLoading(false);
}

document.addEventListener("DOMContentLoaded", () => {
  init().catch(showError);
});
