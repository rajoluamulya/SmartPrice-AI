"""Train the project model using the existing dataset and artifact folders."""

from __future__ import annotations

from pathlib import Path

from src.train import train_and_evaluate


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    metrics = train_and_evaluate(
        dataset_path=PROJECT_ROOT / "data" / "mobile_prices_synthetic.csv",
        model_path=PROJECT_ROOT / "models" / "mobile_price_pipeline.joblib",
        reports_dir=PROJECT_ROOT / "reports",
    )
    print("Held-out test metrics (INR):")
    for name, value in metrics.items():
        print(f"{name.upper():>5}: {value:,.4f}")
    print(f"Saved model to {PROJECT_ROOT / 'models' / 'mobile_price_pipeline.joblib'}")
    print(f"Saved plots and metrics to {PROJECT_ROOT / 'reports'}")


if __name__ == "__main__":
    main()
