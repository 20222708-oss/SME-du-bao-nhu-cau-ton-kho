# Huong dan dau ra mo hinh va bieu do danh gia

Sau khi chay `train` hoac `export`, du an se xuat them cac bang va hinh anh de phuc vu Power BI, bao cao va thuyet trinh.

## Lenh chay

```powershell
python -m retail_forecast.cli export --data "D:/synthetic_data/vn_retail_100sku_3y" --dataset auto --output-dir "D:/retail_artifacts" --max-groups 100
```

Neu chay trong thu muc du an va du lieu nam trong workspace:

```powershell
python -m retail_forecast.cli export --data "synthetic_data/vn_retail_100sku_3y" --dataset auto --output-dir "artifacts" --max-groups 100
```

Mac dinh du an da khai bao `matplotlib` de tao anh PNG. Neu moi truong cua ban chua cai dependency, chay lai lenh cai dat:

```powershell
pip install -e .
```

Neu moi truong khong co `matplotlib`, cac file CSV van duoc xuat binh thuong, chi bo qua viec tao anh PNG.

## Cac file dau ra chinh

| File | Y nghia | Dung trong Power BI |
|---|---|---|
| `fact_history.csv` | Du lieu ban hang lich su da chuan hoa | Duong thuc te, doanh thu, san pham ban chay |
| `fact_forecast.csv` | Du lieu du bao theo ngay va mo hinh | Duong du bao, bang du bao chi tiet |
| `inventory_recommendations.csv` | Safety Stock, Reorder Point, EOQ | Ton kho, canh bao, de xuat dat hang |
| `model_metrics.csv` | So dong du bao, so nhom du bao, ngay forecast | Do phu mo hinh |
| `evaluation_metrics.csv` | Tom tat MAE, RMSE, MAPE/SMAPE, BIAS | KPI danh gia mo hinh |
| `model_evaluation_summary.csv` | Xep hang mo hinh theo sai so | Bang xep hang mo hinh |
| `model_evaluation_detail.csv` | Sai so chi tiet theo ngay, SKU, cua hang, mo hinh | Phan tich sau theo SKU/nhom/thang |
| `model_error_by_category.csv` | Sai so theo nhom san pham | Bieu do nhom san pham nao du bao kho |
| `model_error_by_month.csv` | Sai so theo nam/thang | Heatmap sai so theo thoi gian |

## Cac hinh PNG tu dong sinh

Nam trong thu muc:

```text
charts/
```

Gom:

| File anh | Noi dung |
|---|---|
| `model_smape_ranking.png` | So sanh sai so SMAPE theo mo hinh |
| `model_mae_rmse_comparison.png` | So sanh MAE va RMSE giua cac mo hinh |
| `model_bias_comparison.png` | Xem mo hinh dang du bao cao hon hay thap hon thuc te |
| `forecast_error_distribution.png` | Phan bo sai so du bao |
| `actual_vs_forecast_sample.png` | Thuc te va du bao cua mot SKU tieu bieu |
| `actual_vs_forecast_scatter.png` | Diem du bao co bam sat thuc te hay khong |
| `daily_absolute_error_trend.png` | Ngay nao sai so cao bat thuong |
| `model_error_by_category.png` | Sai so theo nhom san pham |
| `model_error_heatmap_by_month.png` | Heatmap sai so theo thang |
| `top_sku_error.png` | Top SKU co sai so cao can xem lai du lieu/model |

## Power BI nen bieu dien nhu the nao

Trang "Danh gia mo hinh du bao" nen co:

1. Card KPI: MAE, RMSE, MAPE/SMAPE, BIAS.
2. Bang xep hang: Baseline, Prophet, LSTM, Ensemble.
3. Line chart: Thuc te va du bao theo mo hinh.
4. Histogram: Phan bo sai so APE.
5. Bar chart: Sai so theo nhom san pham.
6. Scatter chart: Thuc te so voi du bao.
7. Heatmap: Sai so theo thang.
8. Bang chi tiet: SKU, san pham, cua hang, nhom, model, actual, forecast, error.

## Giai thich khi bao cao

Co the trinh bay ngan gon:

```text
Sau khi train, he thong khong chi xuat ket qua du bao ma con thuc hien backtest tren phan cuoi cua du lieu lich su. Ket qua backtest duoc dung de tinh MAE, RMSE, MAPE/SMAPE va BIAS cho tung mo hinh. Tu do Power BI co the so sanh model, xem nhom san pham nao kho du bao, thang nao sai so cao va SKU nao can cai thien.
```
