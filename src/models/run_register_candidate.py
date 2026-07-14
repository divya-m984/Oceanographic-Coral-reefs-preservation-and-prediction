"""
src/models/run_register_candidate.py — DVC register_candidate stage.

Registers the best model from each task as a new MLflow Model Registry
version (candidate).  Does NOT promote to champion — champion promotion
remains a controlled manual step via src.models.registry --promote.

Writes reports/candidate_registration.json as a DVC-tracked receipt.

Usage
-----
  python -m src.models.run_register_candidate
"""

from __future__ import annotations

import json
import sys

from src.config import get_config, setup_logging
from src.models.registry import run_register_and_promote


def main() -> int:
    logger = setup_logging("coralsense.register_candidate")
    cfg = get_config()

    logger.info("Registering candidates for both tasks (no champion promotion).")

    try:
        results = run_register_and_promote(
            task="all",
            promote=False,  # NEVER promote in the pipeline stage
            cfg=cfg,
        )
    except Exception as exc:
        logger.error("Registration failed: %s", exc, exc_info=True)
        return 1

    receipt: dict = {}
    for task, r in results.items():
        gate_str = "PASS" if r["gate_passed"] else "FAIL"
        receipt[task] = {
            "registered_model_name": r["registered_model_name"],
            "version": str(r["version"]),
            "algo_name": r["algo_name"],
            "cv_macro_f1": round(float(r["cv_macro_f1"]), 6),
            "gate_passed": r["gate_passed"],
            "gate_failures": r["gate_failures"],
            "champion_set": r["champion_set"],
        }
        print(
            f"[{task}] registered v{r['version']} "
            f"algo={r['algo_name']} "
            f"cv_f1={r['cv_macro_f1']:.4f} "
            f"gate={gate_str} "
            f"champion_promoted={r['champion_set']}"
        )

    out = cfg.paths.reports_dir / "candidate_registration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2)

    print(f"Registration receipt → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
