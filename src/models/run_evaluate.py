"""
src/models/run_evaluate.py — DVC evaluate stage for CoralSense.

Reads the per-task evaluation JSON files produced by src.models.train and
writes a single flat DVC metrics file at reports/metrics.json.

Usage
-----
  python -m src.models.run_evaluate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.config import get_config, setup_logging

logger = None


def extract_metrics(models_dir: Path, reports_dir: Path) -> dict:
    """Read evaluation JSONs and return a combined metrics dict."""
    combined: dict = {}
    for task in ("health", "restoration"):
        eval_path = models_dir / f"evaluation_{task}.json"
        if not eval_path.exists():
            raise FileNotFoundError(
                f"Evaluation file not found: {eval_path}. Run src.models.train first."
            )
        with eval_path.open(encoding="utf-8") as fh:
            data = json.load(fh)

        best = data["best_model_name"]
        m = data["models"][best]
        combined[task] = {
            "best_algorithm": best,
            "cv_macro_f1": round(m["cv_macro_f1_mean"], 6),
            "cv_balanced_accuracy": round(m["cv_balanced_accuracy_mean"], 6),
            "test_macro_f1": round(m["test_macro_f1"], 6),
            "test_balanced_accuracy": round(m["test_balanced_accuracy"], 6),
            "test_accuracy": round(m["test_accuracy"], 6),
        }
    return combined


def main() -> int:
    global logger
    logger = setup_logging("coralsense.evaluate")
    cfg = get_config()

    models_dir = cfg.paths.models_dir
    reports_dir = cfg.paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    try:
        metrics = extract_metrics(models_dir, reports_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    out_path = reports_dir / "metrics.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    for task, vals in metrics.items():
        logger.info(
            "[%s] best=%s  cv_macro_f1=%.4f  test_macro_f1=%.4f",
            task,
            vals["best_algorithm"],
            vals["cv_macro_f1"],
            vals["test_macro_f1"],
        )

    print(f"DVC metrics written → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
