from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .exporters import round_numeric_columns
from .models import save_bundle
from .pipeline import train_pipeline
from .synthetic import generate_vn_retail_dataset


def _write_evaluation_metrics(output_dir: Path, metrics: dict[str, dict[str, float | int]]) -> Path:
    rows: list[dict[str, object]] = []
    regression = metrics.get("regression", {})
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

    forecast_models = metrics.get("forecast_models", {})
    coverage = metrics.get("coverage", {})
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

    frame = pd.DataFrame(rows)
    frame = round_numeric_columns(frame)
    path = output_dir / "evaluation_metrics.csv"
    frame.to_csv(path, index=False)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retail-forecast")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train the baseline model")
    train.add_argument("--data", required=True)
    train.add_argument("--dataset", default="auto")
    train.add_argument("--model", default="ridge", choices=["ridge", "ols"])
    train.add_argument("--horizon", type=int, default=30)
    train.add_argument("--output", default="artifacts")
    train.add_argument("--max-groups", type=int, default=25)
    train.add_argument("--no-prophet", action="store_true")
    train.add_argument("--no-lstm", action="store_true")

    forecast = sub.add_parser("forecast", help="Generate forecast")
    forecast.add_argument("--data", required=True)
    forecast.add_argument("--dataset", default="auto")
    forecast.add_argument("--horizon", type=int, default=30)
    forecast.add_argument("--output", default="forecast.csv")
    forecast.add_argument("--max-groups", type=int, default=25)
    forecast.add_argument("--no-prophet", action="store_true")
    forecast.add_argument("--no-lstm", action="store_true")

    export_cmd = sub.add_parser("export", help="Export BI tables for Power BI/Tableau")
    export_cmd.add_argument("--data", required=True)
    export_cmd.add_argument("--dataset", default="auto")
    export_cmd.add_argument("--output-dir", default="artifacts")
    export_cmd.add_argument("--max-groups", type=int, default=25)
    export_cmd.add_argument("--no-prophet", action="store_true")
    export_cmd.add_argument("--no-lstm", action="store_true")

    generate_cmd = sub.add_parser("generate-data", help="Generate a synthetic Vietnamese retail dataset")
    generate_cmd.add_argument("--output-dir", default="synthetic_data/vn_retail")
    generate_cmd.add_argument("--start-date", default="2020-01-01")
    generate_cmd.add_argument("--end-date", default="2024-12-31")
    generate_cmd.add_argument("--num-products", type=int, default=100)
    generate_cmd.add_argument("--num-stores", type=int, default=6)
    generate_cmd.add_argument("--num-suppliers", type=int, default=8)
    generate_cmd.add_argument("--seed", type=int, default=42)

    web_cmd = sub.add_parser("web", help="Run the interactive prediction web app")
    web_cmd.add_argument("--data-root", required=True)
    web_cmd.add_argument("--artifacts-root", default=None)
    web_cmd.add_argument("--host", default="127.0.0.1")
    web_cmd.add_argument("--port", type=int, default=8501)
    web_cmd.add_argument("--reload", action="store_true")
    web_cmd.add_argument("--title", default="Retail Forecast Studio")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        result = train_pipeline(
            args.data,
            profile=args.dataset,
            model_kind=args.model,
            horizon=args.horizon,
            output_dir=args.output,
            forecast_group_limit=args.max_groups,
            enable_prophet=not args.no_prophet,
            enable_lstm=not args.no_lstm,
        )
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_bundle(result.model, str(output_dir / "regression_bundle.joblib"))
        metrics_path = output_dir / "metrics.json"
        pd.Series(result.metrics).to_json(metrics_path)
        _write_evaluation_metrics(output_dir, result.metrics)
        forecast = None
        for key in ("ensemble", "prophet", "lstm", "baseline"):
            candidate = result.forecasts.get(key)
            if candidate is not None and not candidate.empty:
                forecast = candidate
                break
        if forecast is not None:
            forecast = round_numeric_columns(forecast)
            forecast.to_csv(output_dir / "forecast.csv", index=False)
        print(result.metrics)
        return 0

    if args.command == "forecast":
        result = train_pipeline(
            args.data,
            profile=args.dataset,
            horizon=args.horizon,
            forecast_group_limit=args.max_groups,
            enable_prophet=not args.no_prophet,
            enable_lstm=not args.no_lstm,
        )
        forecast = result.forecasts.get("ensemble", result.forecasts.get("baseline", pd.DataFrame()))
        forecast = round_numeric_columns(forecast)
        forecast.to_csv(args.output, index=False)
        print(f"Saved forecast to {args.output}")
        return 0

    if args.command == "export":
        result = train_pipeline(
            args.data,
            profile=args.dataset,
            horizon=30,
            output_dir=args.output_dir,
            forecast_group_limit=args.max_groups,
            enable_prophet=not args.no_prophet,
            enable_lstm=not args.no_lstm,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_evaluation_metrics(output_dir, result.metrics)
        print({k: len(v) for k, v in result.forecasts.items()})
        return 0

    if args.command == "generate-data":
        output_dir = Path(args.output_dir)
        paths = generate_vn_retail_dataset(
            output_dir=output_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            num_products=args.num_products,
            num_stores=args.num_stores,
            num_suppliers=args.num_suppliers,
            seed=args.seed,
        )
        for key, path in paths.items():
            print(f"{key}: {path}")
        return 0

    if args.command == "web":
        from .webapp import run_web_server

        run_web_server(
            data_root=args.data_root,
            artifacts_root=args.artifacts_root,
            host=args.host,
            port=args.port,
            reload=args.reload,
            title=args.title,
        )
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
