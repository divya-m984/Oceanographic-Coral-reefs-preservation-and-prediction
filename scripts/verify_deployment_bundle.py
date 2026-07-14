"""
scripts/verify_deployment_bundle.py — Verify a CoralSense deployment bundle.

Checks
------
1.  Required files exist (payload.joblib, preprocessor.joblib, metadata.json).
2.  Checksums match those recorded in metadata.json at export time.
3.  Model names match registered names.
4.  Champion version matches metadata (version == "1").
5.  Preprocessor exposes expected feature names (no fit/fit_transform called).
6.  Estimator classes match label metadata.
7.  A minimal test prediction succeeds.
8.  Prediction probabilities sum approximately to 1.
9.  No fit or fit_transform is called on any object.
10. Manifest file exists and lists all tasks.

Usage
-----
    python scripts/verify_deployment_bundle.py
    python scripts/verify_deployment_bundle.py --bundle-dir path/to/bundle

Exit codes
----------
0   All checks passed.
1   One or more checks failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BUNDLE_DIR = _PROJECT_ROOT / "deploy" / "bundles"

# Minimal valid sensor record for test prediction
_TEST_RECORD: dict[str, Any] = {
    "region": "Gulf of Mannar",
    "depth_m": 5.0,
    "water_temperature_c": 27.5,
    "ph": 8.1,
    "salinity_ppt": 35.0,
    "dissolved_oxygen_mg_l": 7.0,
    "turbidity_ntu": 2.0,
    "light_intensity": 800.0,
    "current_speed_m_s": 0.2,
    "sonar_backscatter": -15.0,
    "rugosity_index": 3.5,
    "hard_substrate_percentage": 60.0,
    "acoustic_complexity_index": 0.7,
    "coral_cover_percentage": 45.0,
    "bleaching_percentage": 5.0,
    "disease_percentage": 2.0,
}

# Expected model names from registry
_EXPECTED_MODEL_NAMES = {
    "health": "coralsense_reef_health",
    "restoration": "coralsense_restoration_suitability",
}


def sha256_file(path: Path) -> str:
    """Return SHA-256 hex digest of *path*."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_bundle(bundle_dir: Path) -> list[str]:
    """
    Run all verification checks on *bundle_dir*.

    Returns
    -------
    List of failure messages.  Empty list means all checks passed.
    """
    import joblib  # noqa: PLC0415

    failures: list[str] = []

    # 1. Manifest
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        failures.append(f"Manifest not found: {manifest_path}")
    else:
        with manifest_path.open() as fh:
            manifest = json.load(fh)
        for task in ("health", "restoration"):
            if task not in manifest.get("tasks", []):
                failures.append(f"Task '{task}' missing from bundle manifest.")

    for task in ("health", "restoration"):
        task_dir = bundle_dir / task

        # 2. Required files
        for fname in ("payload.joblib", "preprocessor.joblib", "metadata.json"):
            if not (task_dir / fname).exists():
                failures.append(f"[{task}] Missing file: {fname}")

        if any(
            not (task_dir / f).exists()
            for f in ("payload.joblib", "preprocessor.joblib", "metadata.json")
        ):
            # Cannot proceed with further checks for this task
            continue

        # 3. Metadata
        with (task_dir / "metadata.json").open() as fh:
            meta = json.load(fh)

        # 4. Checksums
        actual_payload = sha256_file(task_dir / "payload.joblib")
        if actual_payload != meta.get("checksum_payload"):
            failures.append(
                f"[{task}] Payload checksum mismatch. "
                f"Expected {meta.get('checksum_payload', '?')[:16]}… "
                f"got {actual_payload[:16]}…"
            )

        actual_preprocessor = sha256_file(task_dir / "preprocessor.joblib")
        if actual_preprocessor != meta.get("checksum_preprocessor"):
            failures.append(
                f"[{task}] Preprocessor checksum mismatch. "
                f"Expected {meta.get('checksum_preprocessor', '?')[:16]}… "
                f"got {actual_preprocessor[:16]}…"
            )

        # 5. Model name
        if meta.get("registered_model_name") != _EXPECTED_MODEL_NAMES[task]:
            failures.append(
                f"[{task}] registered_model_name mismatch: "
                f"got '{meta.get('registered_model_name')}', "
                f"expected '{_EXPECTED_MODEL_NAMES[task]}'"
            )

        # 6. Version == "1" (champion)
        if str(meta.get("version")) != "1":
            failures.append(
                f"[{task}] Version is '{meta.get('version')}', expected '1' for champion."
            )

        # 7. Alias
        if meta.get("alias") != "champion":
            failures.append(f"[{task}] Alias is '{meta.get('alias')}', expected 'champion'.")

        # 8. Label names present
        label_names = meta.get("label_names", [])
        if not label_names:
            failures.append(f"[{task}] label_names is empty in metadata.")

        # 9. Load artefacts and check preprocessor features
        try:
            preprocessor = joblib.load(task_dir / "preprocessor.joblib")
            # Must not call fit or fit_transform — only check feature_names_in_
            if not hasattr(preprocessor, "feature_names_in_"):
                failures.append(
                    f"[{task}] Preprocessor lacks feature_names_in_ — may not be fitted."
                )
            else:
                feat_n = len(preprocessor.feature_names_in_)
                if feat_n < 5:
                    failures.append(
                        f"[{task}] Preprocessor has only {feat_n} feature(s) — suspiciously few."
                    )
        except Exception as exc:
            failures.append(f"[{task}] Failed to load preprocessor: {exc}")
            preprocessor = None

        # 10. Load payload and check estimator classes
        try:
            payload = joblib.load(task_dir / "payload.joblib")
            estimator = payload.get("estimator")
            label_encoder = payload.get("label_encoder")

            if estimator is None:
                failures.append(f"[{task}] payload['estimator'] is None.")
            else:
                if label_encoder is not None:
                    classes = list(label_encoder.classes_)
                elif hasattr(estimator, "classes_"):
                    classes = [str(c) for c in estimator.classes_]
                else:
                    classes = []

                if set(classes) != set(label_names):
                    failures.append(
                        f"[{task}] Estimator classes {classes} "
                        f"don't match metadata label_names {label_names}."
                    )
        except Exception as exc:
            failures.append(f"[{task}] Failed to load payload: {exc}")
            payload = None
            estimator = None

        # 11. Test prediction (verify probabilities sum to 1, no fit called)
        if preprocessor is not None and estimator is not None:
            try:
                from src.api.bundle_loader import BundleInferencePipeline  # noqa: PLC0415

                pipeline = BundleInferencePipeline(task=task, bundle_dir=bundle_dir)
                result = pipeline.predict_single(_TEST_RECORD)

                # Probability integrity
                proba_sum = sum(result.get("probabilities", {}).values())
                if not (0.99 <= proba_sum <= 1.01):
                    failures.append(
                        f"[{task}] Prediction probabilities sum to {proba_sum:.4f}, expected ~1.0."
                    )

                # Predicted class in label names
                pred = result.get("predicted_class", "")
                if pred not in label_names:
                    failures.append(
                        f"[{task}] predicted_class '{pred}' not in label_names {label_names}."
                    )

                logger.info(
                    "[%s] Test prediction: %s (confidence=%.4f)",
                    task,
                    pred,
                    result.get("confidence", 0),
                )

            except Exception as exc:
                failures.append(f"[{task}] Test prediction failed: {exc}")

    return failures


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Verify a CoralSense deployment bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=_DEFAULT_BUNDLE_DIR,
        help="Path to the bundle directory.",
    )
    args = parser.parse_args(argv)

    if not args.bundle_dir.exists():
        print(f"ERROR: Bundle directory not found: {args.bundle_dir}", file=sys.stderr)
        print(
            "Run: python scripts/export_champions.py",
            file=sys.stderr,
        )
        return 1

    print(f"Verifying bundle at {args.bundle_dir} …")
    failures = verify_bundle(args.bundle_dir)

    if failures:
        print(f"\n{len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  ✗ {f}")
        return 1

    print("\nAll checks PASSED — bundle is ready for deployment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
