# Hệ thống dự báo nhu cầu và tối ưu tồn kho bán lẻ

Dự án này phục vụ đồ án tốt nghiệp với mục tiêu xây dựng một hệ thống hỗ trợ doanh nghiệp bán lẻ vừa và nhỏ:

- Phân tích dữ liệu bán hàng lịch sử.
- Dự báo nhu cầu sản phẩm theo ngày.
- So sánh các mô hình dự báo.
- Tính toán tồn kho an toàn, điểm đặt hàng lại và EOQ.
- Xuất dữ liệu phục vụ dashboard Power BI.
- Cung cấp giao diện desktop Python để người dùng thao tác trực tiếp.

Hệ thống đóng vai trò hỗ trợ ra quyết định. Kết quả dự báo và khuyến nghị nhập hàng cần được người quản lý xem xét cùng với kinh nghiệm thực tế, sức chứa kho, ngân sách nhập hàng và chính sách nhà cung cấp.

## Công nghệ sử dụng

- Python: ngôn ngữ xử lý chính cho pipeline dữ liệu, mô hình dự báo và tính toán tồn kho.
- Pandas, NumPy: tiền xử lý dữ liệu, tổng hợp bảng và tính toán chỉ số.
- Regression baseline: mô hình nền để so sánh.
- Prophet: mô hình chuỗi thời gian có khả năng xử lý xu hướng, mùa vụ và ngày lễ.
- TensorFlow/Keras LSTM: mô hình học sâu cho chuỗi thời gian.
- Ensemble: kết hợp kết quả Prophet và LSTM.
- Power BI: trực quan hóa KPI, xu hướng, tồn kho và so sánh mô hình.
- Tkinter: giao diện desktop Python.
- FastAPI/Streamlit: các phiên bản giao diện web thử nghiệm.

## Cấu trúc dự án

```text
.
├── src/retail_forecast/
│   ├── cli.py              # Lệnh train, forecast, export, generate-data, desktop, web, ui
│   ├── datasets.py         # Chuẩn hóa và đọc dữ liệu
│   ├── features.py         # Tạo đặc trưng thời gian, lag, rolling
│   ├── models.py           # Regression, Prophet, LSTM, Ensemble
│   ├── inventory.py        # Safety Stock, Reorder Point, EOQ
│   ├── pipeline.py         # Điều phối quá trình train/export
│   ├── exporters.py        # Xuất bảng CSV cho Power BI
│   ├── desktop_app.py      # Giao diện desktop Python
│   ├── streamlit_app.py    # Giao diện Streamlit
│   ├── webapp.py           # API và web dashboard
│   └── synthetic.py        # Sinh bộ dữ liệu synthetic retail Việt Nam
├── tests/                  # Kiểm thử pipeline và web repository
├── web/                    # Giao diện web tĩnh
├── pyproject.toml          # Cấu hình package và dependencies
└── README.md
```

## Cài đặt môi trường

Yêu cầu Python từ 3.10 trở lên.

```bash
pip install -e .
```

Nếu cần chạy đầy đủ Prophet, LSTM và giao diện:

```bash
pip install -e ".[prophet,lstm,ui,web,dev]"
```

Trên Windows PowerShell, nếu dấu ngoặc gây lỗi, vẫn dùng cú pháp:

```powershell
pip install -e ".[prophet,lstm,ui,web,dev]"
```

## Sinh bộ dữ liệu synthetic

Lệnh sinh bộ dữ liệu bán lẻ Việt Nam 100 sản phẩm:

```bash
retail-forecast generate-data --output-dir synthetic_data/vn_retail_100sku_3y --start-date 2022-01-01 --end-date 2024-12-31
```

Bộ dữ liệu synthetic có các bảng phục vụ Power BI và mô hình dự báo, gồm:

- `dim_product.csv`
- `dim_store.csv`
- `dim_calendar.csv`
- `dim_supplier.csv`
- `fact_sales.csv`
- `fact_inventory.csv`
- `fact_promotion.csv`
- `fact_purchase_orders.csv`
- `fact_returns.csv`
- `fact_weather.csv`

## Train mô hình

Chạy train với dữ liệu synthetic:

```bash
retail-forecast train --data synthetic_data/vn_retail_100sku_3y --dataset auto --output artifacts --max-groups 100
```

Trong đó:

- `--data`: thư mục dữ liệu đầu vào.
- `--dataset auto`: tự động nhận diện và chuẩn hóa dữ liệu.
- `--output`: thư mục lưu kết quả train.
- `--max-groups`: số chuỗi `store_id + item_id` được xử lý.

Nếu muốn chạy nhanh để kiểm thử:

```bash
retail-forecast train --data synthetic_data/vn_retail_100sku_3y --dataset auto --output artifacts --max-groups 10 --no-prophet --no-lstm
```

## Xuất dữ liệu cho Power BI

Sau khi train, xuất các bảng phục vụ dashboard:

```bash
retail-forecast export --data synthetic_data/vn_retail_100sku_3y --dataset auto --output-dir artifacts --max-groups 100
```

Thư mục `artifacts` sẽ có các file chính:

- `fact_history.csv`: dữ liệu bán hàng lịch sử đã chuẩn hóa.
- `fact_forecast.csv`: dữ liệu dự báo theo ngày.
- `inventory_recommendations.csv`: khuyến nghị tồn kho và đặt hàng.
- `model_metrics.csv`: số dòng và độ phủ của các mô hình.
- `evaluation_metrics.csv`: chỉ số MAE, RMSE, MAPE, BIAS nếu có.
- `dim_product.csv`: danh mục sản phẩm.
- `dim_store.csv`: danh mục cửa hàng.

## Chạy giao diện desktop Python

Nếu kết quả đã được xuất vào `D:/retail_artifacts`:

```bash
python -m retail_forecast.cli desktop --data-root D:/retail_artifacts
```

Nếu dùng console script sau khi cài đặt package:

```bash
retail-forecast desktop --data-root D:/retail_artifacts
```

Giao diện desktop hỗ trợ:

- Xem tổng quan bán hàng.
- Lọc theo kho, nhóm sản phẩm, sản phẩm và mô hình.
- Xem biểu đồ dự báo theo ngày.
- Xem tồn kho hiện tại, tồn kho an toàn và đề xuất đặt hàng.
- Xem bảng so sánh mô hình.
- Kiểm tra các bảng dữ liệu đầu vào và đầu ra.

## Chạy giao diện Streamlit

```bash
retail-forecast ui --data-root D:/retail_artifacts
```

## Chạy web dashboard FastAPI

```bash
retail-forecast web --data-root D:/retail_artifacts --host 127.0.0.1 --port 8000
```

Sau đó mở trình duyệt tại:

```text
http://127.0.0.1:8000
```

## Công thức tồn kho

Tồn kho an toàn:

```text
Safety Stock = Z * demand_std * sqrt(lead_time)
```

Điểm đặt hàng lại:

```text
Reorder Point = avg_demand * lead_time + Safety Stock
```

Lượng đặt hàng tối ưu:

```text
EOQ = sqrt((2 * D * S) / H)
```

Trong đó:

- `avg_demand`: nhu cầu trung bình.
- `demand_std`: độ lệch chuẩn nhu cầu.
- `lead_time`: thời gian từ lúc đặt hàng đến lúc hàng về kho.
- `Z`: hệ số mức độ phục vụ.
- `D`: nhu cầu trong kỳ.
- `S`: chi phí mỗi lần đặt hàng.
- `H`: chi phí lưu kho một đơn vị.

## Kiểm thử

```bash
pip install -e ".[dev]"
pytest -q
```

## Lưu ý triển khai

- Dữ liệu synthetic dùng để mô phỏng và kiểm thử pipeline, chưa thay thế hoàn toàn dữ liệu thật.
- Khi áp dụng thực tế cần cấu hình lại lead time, service level, chi phí đặt hàng và chi phí lưu kho.
- Kết quả dự báo có sai số, nên được xem là cơ sở hỗ trợ ra quyết định thay vì quyết định tự động tuyệt đối.
