# Dự án Dự Báo Nhu Cầu Bán Lẻ

Dự án này là bộ khung tái sử dụng cho bài toán **dự báo nhu cầu bán lẻ** và **tối ưu hóa tồn kho**.

Dự án được thiết kế để xử lý nhiều bộ dữ liệu nhìn khác nhau bên ngoài, ví dụ như:

- M5 Forecasting - Accuracy
- Rossmann Store Sales
- Walmart Sales Forecasting
- Corporacion Favorita Grocery Sales Forecasting
- Store Sales - Time Series Forecasting

Điểm mấu chốt là chuẩn hóa mọi bộ dữ liệu về một schema chung:

- `date`
- `store_id`
- `item_id`
- `target`
- các trường ngoại sinh tùy chọn như `promo`, `holiday`, `price`, `transactions`, `oil`, `cpi`, `unemployment`

## Có gì trong dự án

- Chuẩn hóa dữ liệu đầu vào
- Bộ chuyển đổi sẵn cho các dataset bán lẻ như M5, Rossmann, Walmart, Favorita và Store Sales
- Tạo đặc trưng chuỗi thời gian
- Mô hình hồi quy baseline
- Prophet tùy chọn
- LSTM tùy chọn
- Các chỉ số tồn kho: safety stock, reorder point, EOQ
- Xuất CSV sẵn cho Power BI/Tableau

## Cài đặt

```bash
pip install -e .
```

Nếu muốn dùng thêm mô hình tùy chọn:

```bash
pip install -e ".[prophet]"
pip install -e ".[lstm]"
```

## Huấn luyện mô hình

```bash
retail-forecast train --data path/to/train.csv --dataset auto --output artifacts
```

Mặc định pipeline sẽ dự báo theo từng nhóm quan trọng trước để chạy nhẹ trên máy local.
Bạn có thể dùng `--max-groups` để tăng hoặc giảm số chuỗi `store_id + item_id` được xử lý.

## Dự báo

```bash
retail-forecast forecast --data path/to/train.csv --dataset auto --horizon 30 --output forecasts.csv
```

## Xuất bảng cho Power BI / Tableau

```bash
retail-forecast export --data path/to/data_or_folder --dataset auto --output-dir artifacts
```

Lệnh này sẽ tạo các bảng đầu ra phù hợp để làm dashboard:

- `fact_history.csv`
- `fact_forecast.csv`
- `dim_product.csv`
- `dim_store.csv`
- `inventory_recommendations.csv`
- `model_metrics.csv`

## Quy trình sử dụng mẫu

1. Đặt dữ liệu vào `data/raw/`
2. Chuẩn hóa dữ liệu
3. Huấn luyện regression / Prophet / LSTM
4. Sinh kết quả dự báo
5. Tính khuyến nghị tồn kho
6. Xuất CSV để làm dashboard

## Vì sao cách này dùng được cho nhiều bộ dữ liệu

Các bộ dữ liệu không giống hệt nhau, nhưng đều mô tả nhu cầu bán lẻ theo thời gian. Khi chuẩn hóa về cùng schema, ta có thể xây dựng một pipeline chung để:

- ánh xạ tên cột về chuẩn thống nhất
- nhận biết tần suất thời gian
- tạo lag và rolling statistics
- huấn luyện trên cùng định nghĩa target

Như vậy, bạn không cần làm 5 dự án riêng. Chỉ cần một pipeline có bộ chuyển đổi phù hợp.
