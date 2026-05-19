from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .exporters import round_numeric_columns
from .models import save_bundle
from .pipeline import train_pipeline


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
        print({k: len(v) for k, v in result.forecasts.items()})
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
