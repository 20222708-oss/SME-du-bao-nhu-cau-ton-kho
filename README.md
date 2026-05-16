# Retail Forecast Project

This project is a reusable starter for retail demand forecasting and inventory optimization.

It is designed to handle datasets that look different on the surface, such as:

- M5 Forecasting - Accuracy
- Rossmann Store Sales
- Walmart Sales Forecasting
- Corporacion Favorita Grocery Sales Forecasting
- Store Sales - Time Series Forecasting

The trick is to normalize each dataset into a shared internal schema:

- `date`
- `store_id`
- `item_id`
- `target`
- optional exogenous fields like `promo`, `holiday`, `price`, `transactions`, `oil`, `cpi`, `unemployment`

## What is included

- Dataset normalization
- Built-in adapters for retail datasets like M5, Rossmann, Walmart, Favorita, and Store Sales
- Time-series feature engineering
- Baseline regression model
- Optional Prophet forecasting
- Optional LSTM forecaster
- Inventory metrics: safety stock, reorder point, EOQ
- Power BI/Tableau-ready CSV exports

## Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[prophet]"
pip install -e ".[lstm]"
```

## Train a model

```bash
retail-forecast train --data path/to/train.csv --dataset auto --output artifacts
```

## Forecast

```bash
retail-forecast forecast --data path/to/train.csv --dataset auto --horizon 30 --output forecasts.csv
```

## Export BI tables

```bash
retail-forecast export --data path/to/data_or_folder --dataset auto --output-dir artifacts
```

This generates BI-friendly tables:

- `fact_history.csv`
- `fact_forecast.csv`
- `dim_product.csv`
- `dim_store.csv`
- `inventory_recommendations.csv`
- `model_metrics.csv`

## Example workflow

1. Drop a dataset into `data/raw/`
2. Run normalization
3. Train regression / Prophet / LSTM
4. Generate a forecast
5. Compute inventory recommendations
6. Export results to CSV for a dashboard

## Why this approach works for different datasets

The datasets are not identical, but they all describe retail demand over time. A shared schema lets us build one pipeline that:

- maps column names to a standard form
- detects the time granularity
- creates lags and rolling statistics
- trains models on the same target definition

That means you do not need five separate projects. You need one pipeline with adapters.
