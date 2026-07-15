"""
scripts/run_retraining.py — CoralSense retraining orchestrator CLI.

Chains: validate input → check drift → retrain challengers → compare vs champion.
Promotion and rollback are separate explicit steps (promote.py / rollback.py).

USAGE
-----
  # Full run with drift summary (recommended path)
  python scripts/run_retraining.py \\
    --input data/raw/observations_validated.csv \\
    --task all \\
    --data-source synthetic \\
    --drift-summary reports/drift_summary.json \\
    --quick

  # Dry-run: validate only, no training
  python scripts/run_retraining.py \\
    --input data/raw/observations_validated.csv \\
    --task all \\
    --data-source synthetic \\
    --drift-summary reports/drift_summary.json \\
    --dry-run

  # Manual reason (no drift summary required)
  python scripts/run_retraining.py \\
    --input labelled_field_data.csv \\
    --task health \\
    --data-source field_labelled \\
    --reason "Post-bleaching-event field survey, July 2026" \\
    --quick

NOTE: This script does NOT promote any model to champion.
      Use:  python -m src.models.promote  to promote after review.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """
    Orchestrate retraining and (optionally) comparison against current champion.

    Exit codes
    ----------
    0   All requested steps completed.
    1   Error in one or more steps.
    2   Invalid arguments.
    3   Permission denied.
    """
    from src.config import get_config, setup_logging
    from src.models.compare import compare_challenger
    from src.models.retrain import run_retraining

    setup_logging("coralsense.run_retraining")

    parser = argparse.ArgumentParser(
        description="CoralSense retraining orchestrator (retrain + compare).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        metavar="CSV",
        help="Path to labelled observations CSV.",
    )
    parser.add_argument(
        "--task",
        choices=["health", "restoration", "all"],
        default="all",
        help="Which task(s) to retrain.",
    )
    parser.add_argument(
        "--data-source",
        required=True,
        choices=["synthetic", "field_labelled"],
        dest="data_source",
        help="Declare data source.",
    )
    parser.add_argument(
        "--drift-summary",
        type=Path,
        default=None,
        dest="drift_summary",
        metavar="JSON",
        help="Path to drift_summary.json.",
    )
    parser.add_argument(
        "--reason",
        default=None,
        metavar="TEXT",
        help="Manual retraining reason (if no drift summary).",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=None,
        dest="mlflow_uri",
        metavar="URI",
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        dest="output_dir",
        metavar="DIR",
        help="Directory for receipt and comparison JSONs.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="Quick mode: reduced CV folds and estimators (for CI / tests).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help="Validate and check permission only — no training, no registry changes.",
    )
    parser.add_argument(
        "--skip-compare",
        action="store_true",
        default=False,
        dest="skip_compare",
        help="Skip champion–challenger comparison after retraining.",
    )
    args = parser.parse_args(argv)

    tasks = ["health", "restoration"] if args.task == "all" else [args.task]

    try:
        cfg = get_config()
        mlflow_uri = args.mlflow_uri or cfg.mlflow_tracking_uri
        output_dir = args.output_dir or cfg.paths.reports_dir

        # ── Step 1: Retraining ───────────────────────────────────────────────
        print(f"\n{'=' * 60}")
        print("CoralSense Retraining Orchestrator")
        print(f"  Input     : {args.input}")
        print(f"  Tasks     : {tasks}")
        print(f"  Source    : {args.data_source}")
        print(f"  Quick     : {args.quick}")
        print(f"  Dry-run   : {args.dry_run}")
        print(f"{'=' * 60}\n")

        receipt = run_retraining(
            input_path=args.input,
            tasks=tasks,
            data_source=args.data_source,
            drift_summary_path=args.drift_summary,
            manual_reason=args.reason,
            mlflow_uri=mlflow_uri,
            output_dir=output_dir,
            cfg=cfg,
            quick=args.quick,
            dry_run=args.dry_run,
        )

        if receipt.dry_run:
            print("\nDRY-RUN complete. No models trained or registered.")
            print(f"  Input rows : {receipt.n_rows}")
            print(f"  Input SHA  : {receipt.input_sha256[:16]}...")
            print(f"\nReceipt ID : {receipt.receipt_id}")
            return 0

        print(f"\nRetraining complete. Receipt ID: {receipt.receipt_id}")
        for ch in receipt.challengers:
            print(
                f"  [{ch['task']}] {ch['registered_model_name']} v{ch['version']} "
                f"algo={ch['algo_name']} cv_f1={ch['cv_macro_f1']:.4f}"
            )

        # ── Step 2: Compare (optional) ───────────────────────────────────────
        if args.skip_compare:
            print("\nComparison skipped (--skip-compare).")
            print("Run: python -m src.models.compare --task <task> --challenger-version <version>")
            return 0

        print(f"\n{'─' * 60}")
        print("Champion–Challenger Comparison")
        print(f"{'─' * 60}")

        all_eligible = True
        for ch in receipt.challengers:
            task = ch["task"]
            version = ch["version"]
            compare_output = output_dir / f"compare_{task}_{receipt.receipt_id}.json"

            try:
                report = compare_challenger(
                    task=task,
                    challenger_version=version,
                    mlflow_uri=mlflow_uri,
                    output_path=compare_output,
                    cfg=cfg,
                )
                print(f"\n[{task}] Outcome: {report.outcome.upper()}")
                print(
                    f"  Champion  : v{report.champion['version']} "
                    f"cv_f1={report.champion['cv_macro_f1']:.4f}"
                )
                print(
                    f"  Challenger: v{report.challenger['version']} "
                    f"cv_f1={report.challenger['cv_macro_f1']:.4f}"
                )
                if report.failures:
                    print("  Failures:")
                    for f in report.failures:
                        print(f"    - {f}")
                if report.warnings:
                    print("  Warnings:")
                    for w in report.warnings:
                        print(f"    ! {w}")
                print(f"  Report saved: {compare_output}")

                if report.outcome != "eligible_for_promotion":
                    all_eligible = False
            except Exception as exc:
                logger.error("[%s] Comparison failed: %s", task, exc)
                all_eligible = False

        print(f"\n{'=' * 60}")
        if all_eligible:
            print("All challengers: ELIGIBLE FOR PROMOTION")
            print("\nTo promote, run:")
            for ch in receipt.challengers:
                model = ch["registered_model_name"]
                ver = ch["version"]
                print(
                    f"  python -m src.models.promote "
                    f"--model {model} --version {ver} "
                    f"--comparison-report <report.json> "
                    f"--approve --approver 'Name' --reason 'Reason'"
                )
        else:
            print("One or more challengers require review or were rejected.")
            print("Champion aliases remain UNCHANGED.")
        print(f"{'=' * 60}\n")

        return 0

    except PermissionError as exc:
        logger.error("Permission denied: %s", exc)
        return 3
    except Exception as exc:
        logger.error("Orchestration failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
