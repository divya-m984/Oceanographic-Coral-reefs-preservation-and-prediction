"""
src/models/registry.py — MLflow Model Registry for CoralSense.

Responsibilities
----------------
- Register the best-candidate model for each task into the MLflow Model Registry.
- Attach rich metadata tags (algo, metrics, features, labels, timestamp, git hash).
- Enforce configurable quality gates (min CV macro-F1, min CV balanced-accuracy)
  before promoting any candidate to the "champion" alias.
- Promote the passing candidate by setting the champion alias.
- Provide helpers to resolve the current champion version.

SYNTHETIC-DATA DISCLAIMER
--------------------------
All metrics reflect performance on the CoralSense *synthetic* dataset.
They must NOT be interpreted as evidence of real-world conservation accuracy.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlflow import MlflowClient

from src.config import Config, get_config, setup_logging

logger = logging.getLogger(__name__)

_SYNTHETIC_DISCLAIMER = (
    "Metrics reflect performance on synthetic data only. "
    "Do not use to infer real-world conservation accuracy."
)

_VALID_TASKS = ("health", "restoration")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_hash() -> str:
    """Return the current HEAD commit hash, or 'unknown' if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _registered_model_name(task: str, cfg: Config) -> str:
    if task == "health":
        return cfg.mlflow_registered_health
    return cfg.mlflow_registered_restoration


def _experiment_name(task: str, cfg: Config) -> str:
    if task == "health":
        return cfg.mlflow_experiment_health
    return cfg.mlflow_experiment_restoration


def _resolve_tracking_uri(mlflow_uri: str | None, cfg: Config) -> str:
    return mlflow_uri or cfg.mlflow_tracking_uri


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class QualityGateResult:
    """Outcome of a quality-gate check."""

    task: str
    passed: bool
    cv_macro_f1: float
    min_cv_macro_f1: float
    cv_balanced_accuracy: float
    min_cv_balanced_accuracy: float
    failures: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"QualityGate[{self.task}] {status} — "
            f"cv_macro_f1={self.cv_macro_f1:.4f} (min={self.min_cv_macro_f1:.4f}), "
            f"cv_bal_acc={self.cv_balanced_accuracy:.4f} (min={self.min_cv_balanced_accuracy:.4f})"
        )


@dataclass
class RegistrationResult:
    """Information about a newly registered model version."""

    task: str
    registered_model_name: str
    version: str
    run_id: str
    algo_name: str
    cv_macro_f1: float
    gate: QualityGateResult
    champion_set: bool = False


# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------


def check_quality_gates(
    task: str,
    cv_macro_f1: float,
    cv_balanced_accuracy: float,
    cfg: Config,
) -> QualityGateResult:
    """
    Check whether a candidate model meets the promotion thresholds.

    Uses CV metrics only — never final test-set performance — to avoid
    selecting models on held-out data.

    Parameters
    ----------
    task:
        ``"health"`` or ``"restoration"``.
    cv_macro_f1:
        Cross-validated macro F1 score of the candidate.
    cv_balanced_accuracy:
        Cross-validated balanced accuracy of the candidate.
    cfg:
        Config instance containing quality_gates section.

    Returns
    -------
    QualityGateResult with ``passed`` flag and per-metric breakdown.
    """
    gates = cfg.quality_gates.get(task, {})
    min_f1 = float(gates.get("min_cv_macro_f1", 0.0))
    min_bal = float(gates.get("min_cv_balanced_accuracy", 0.0))

    failures: list[str] = []
    if cv_macro_f1 < min_f1:
        failures.append(f"cv_macro_f1 {cv_macro_f1:.4f} < required {min_f1:.4f}")
    if cv_balanced_accuracy < min_bal:
        failures.append(f"cv_balanced_accuracy {cv_balanced_accuracy:.4f} < required {min_bal:.4f}")

    return QualityGateResult(
        task=task,
        passed=len(failures) == 0,
        cv_macro_f1=cv_macro_f1,
        min_cv_macro_f1=min_f1,
        cv_balanced_accuracy=cv_balanced_accuracy,
        min_cv_balanced_accuracy=min_bal,
        failures=failures,
    )


# ---------------------------------------------------------------------------
# Model version registration
# ---------------------------------------------------------------------------


def _get_or_create_registered_model(
    client: MlflowClient,
    model_name: str,
    description: str,
) -> None:
    """Create the registered model if it does not already exist."""
    try:
        client.get_registered_model(model_name)
        logger.debug("Registered model '%s' already exists.", model_name)
    except Exception:
        client.create_registered_model(
            name=model_name,
            description=description,
        )
        logger.info("Created registered model '%s'.", model_name)


def _find_logged_model_artifact_location(
    client: MlflowClient,
    experiment_id: str,
    run_id: str,
) -> str | None:
    """Return the artifact_location for the LoggedModel attached to *run_id*."""
    try:
        logged_models = client.search_logged_models(experiment_ids=[experiment_id])
        for lm in logged_models:
            if lm.source_run_id == run_id:
                return lm.artifact_location
    except Exception as exc:
        logger.warning("Could not search logged models: %s", exc)
    return None


def register_candidate(
    task: str,
    mlflow_uri: str | None = None,
    output_dir: Path | None = None,
    cfg: Config | None = None,
) -> RegistrationResult:
    """
    Register the best candidate model for *task* in the MLflow Model Registry.

    Reads the evaluation JSON produced by ``train_task`` to identify:
    - The best run (by CV macro-F1, as selected during training).
    - The algorithm name and all metrics.

    Tags the model version with rich metadata including feature names,
    label names, training timestamp, git hash, and synthetic-data disclaimer.

    Does NOT promote to champion — call :func:`promote_champion` separately.

    Parameters
    ----------
    task:
        ``"health"`` or ``"restoration"``.
    mlflow_uri:
        SQLite tracking URI.  Falls back to ``cfg.mlflow_tracking_uri``.
    output_dir:
        Directory containing ``best_model_{task}.joblib`` and
        ``evaluation_{task}.json``.  Defaults to ``cfg.paths.models_dir``.
    cfg:
        Config instance.

    Returns
    -------
    RegistrationResult
    """
    if task not in _VALID_TASKS:
        raise ValueError(f"task must be one of {_VALID_TASKS}, got '{task}'")

    cfg = cfg or get_config()
    tracking_uri = _resolve_tracking_uri(mlflow_uri, cfg)
    output_dir = output_dir or cfg.paths.models_dir

    # Load evaluation JSON
    eval_path = output_dir / f"evaluation_{task}.json"
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {eval_path}. Run src.models.train first."
        )
    with eval_path.open(encoding="utf-8") as fh:
        eval_summary = json.load(fh)

    best_name = eval_summary["best_model_name"]
    best_metrics = eval_summary["models"][best_name]
    run_id: str = best_metrics["mlflow_run_id"]
    label_names: list[str] = eval_summary["label_names"]
    cv_macro_f1: float = best_metrics["cv_macro_f1_mean"]
    cv_bal_acc: float = best_metrics["cv_balanced_accuracy_mean"]

    # Check quality gate (informational at registration time)
    gate = check_quality_gates(task, cv_macro_f1, cv_bal_acc, cfg)

    client = MlflowClient(tracking_uri=tracking_uri)
    model_name = _registered_model_name(task, cfg)
    exp_name = _experiment_name(task, cfg)

    # Resolve experiment_id for logged-model lookup
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        raise RuntimeError(f"MLflow experiment '{exp_name}' not found. Run src.models.train first.")

    # Locate the MLflow logged-model artifact source
    artifact_location = _find_logged_model_artifact_location(client, exp.experiment_id, run_id)
    if artifact_location is None:
        raise RuntimeError(
            f"Could not find a LoggedModel artifact for run {run_id}. "
            "Ensure train.py logged the model via mlflow.sklearn/xgboost.log_model."
        )

    # Get or create the registered model entry
    task_label = "reef health" if task == "health" else "restoration suitability"
    _get_or_create_registered_model(
        client,
        model_name,
        description=(
            f"CoralSense {task_label} classifier. "
            f"Best algorithm selected by CV macro-F1. "
            f"{_SYNTHETIC_DISCLAIMER}"
        ),
    )

    # Collect rich tags
    run_obj = client.get_run(run_id)
    feature_names_str = run_obj.data.params.get("feature_names", "")
    # If not stored as a param (old runs), derive from n_features count
    if not feature_names_str:
        n_feat = run_obj.data.params.get("n_features", "")
        feature_names_str = f"{n_feat} features (names not stored in params)"

    joblib_path = str(output_dir / f"best_model_{task}.joblib")

    tags: dict[str, str] = {
        "task": task,
        "algo_name": best_name,
        "run_id": run_id,
        "label_names": ",".join(label_names),
        "cv_macro_f1": f"{cv_macro_f1:.6f}",
        "cv_balanced_accuracy": f"{cv_bal_acc:.6f}",
        "test_macro_f1": f"{best_metrics['test_macro_f1']:.6f}",
        "test_accuracy": f"{best_metrics['test_accuracy']:.6f}",
        "test_balanced_accuracy": f"{best_metrics['test_balanced_accuracy']:.6f}",
        "train_rows": run_obj.data.params.get("train_rows", ""),
        "n_features": run_obj.data.params.get("n_features", ""),
        "feature_names": feature_names_str,
        "training_timestamp": datetime.now(UTC).isoformat(),
        "git_commit": _git_hash(),
        "joblib_path": joblib_path,
        "synthetic_data_disclaimer": _SYNTHETIC_DISCLAIMER,
        "quality_gate_passed": str(gate.passed),
        "quality_gate_min_cv_macro_f1": f"{gate.min_cv_macro_f1:.6f}",
        "quality_gate_min_cv_balanced_accuracy": f"{gate.min_cv_balanced_accuracy:.6f}",
    }

    # Create model version
    mv = client.create_model_version(
        name=model_name,
        source=artifact_location,
        run_id=run_id,
        tags=tags,
        description=(
            f"algo={best_name} | CV macro-F1={cv_macro_f1:.4f} | "
            f"gate={'PASS' if gate.passed else 'FAIL'}"
        ),
    )
    # MLflow 3.x returns version as int; normalise to str for consistency.
    version_str = str(mv.version)

    logger.info(
        "[%s] Registered '%s' version %s (run=%s, algo=%s, CV macro-F1=%.4f, gate=%s)",
        task,
        model_name,
        version_str,
        run_id[:8],
        best_name,
        cv_macro_f1,
        "PASS" if gate.passed else "FAIL",
    )

    return RegistrationResult(
        task=task,
        registered_model_name=model_name,
        version=version_str,
        run_id=run_id,
        algo_name=best_name,
        cv_macro_f1=cv_macro_f1,
        gate=gate,
    )


# ---------------------------------------------------------------------------
# Champion promotion
# ---------------------------------------------------------------------------


def promote_champion(
    task: str,
    candidate_version: str,
    mlflow_uri: str | None = None,
    cfg: Config | None = None,
) -> str:
    """
    Promote a registered model version to the champion alias.

    Verifies that the version has passed quality gates (stored as a tag).
    Refuses promotion if the quality gate was not passed.

    Parameters
    ----------
    task:
        ``"health"`` or ``"restoration"``.
    candidate_version:
        Model version string (e.g. ``"1"``).
    mlflow_uri:
        SQLite tracking URI.
    cfg:
        Config instance.

    Returns
    -------
    The champion alias string (e.g. ``"champion"``).
    """
    if task not in _VALID_TASKS:
        raise ValueError(f"task must be one of {_VALID_TASKS}, got '{task}'")

    cfg = cfg or get_config()
    tracking_uri = _resolve_tracking_uri(mlflow_uri, cfg)
    client = MlflowClient(tracking_uri=tracking_uri)
    model_name = _registered_model_name(task, cfg)
    alias = cfg.mlflow_champion_alias

    # Fetch and verify quality gate tag
    mv = client.get_model_version(name=model_name, version=candidate_version)
    # MLflow 3.x returns tags as a plain dict; older versions returned a list of tag objects.
    raw_tags = mv.tags or {}
    tags_dict = raw_tags if isinstance(raw_tags, dict) else {t.key: t.value for t in raw_tags}
    gate_passed = tags_dict.get("quality_gate_passed", "False").lower() == "true"

    if not gate_passed:
        cv_f1 = tags_dict.get("cv_macro_f1", "unknown")
        min_f1 = tags_dict.get("quality_gate_min_cv_macro_f1", "unknown")
        raise RuntimeError(
            f"Cannot promote version {candidate_version} of '{model_name}' to "
            f"'{alias}': quality gate not passed "
            f"(cv_macro_f1={cv_f1} < min={min_f1})."
        )

    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=candidate_version,
    )
    logger.info(
        "[%s] Promoted version %s of '%s' to alias '%s'.",
        task,
        candidate_version,
        model_name,
        alias,
    )
    return alias


# ---------------------------------------------------------------------------
# Champion resolution
# ---------------------------------------------------------------------------


def get_champion_version(
    task: str,
    mlflow_uri: str | None = None,
    cfg: Config | None = None,
) -> dict[str, Any]:
    """
    Resolve the current champion model version for *task*.

    Returns
    -------
    Dict with:
        - ``registered_model_name``
        - ``version``
        - ``run_id``
        - ``algo_name``
        - ``task``
        - ``label_names``  (list)
        - ``cv_macro_f1``
        - ``joblib_path``
        - ``alias``
        - ``creation_timestamp``
        - all raw tags
    """
    if task not in _VALID_TASKS:
        raise ValueError(f"task must be one of {_VALID_TASKS}, got '{task}'")

    cfg = cfg or get_config()
    tracking_uri = _resolve_tracking_uri(mlflow_uri, cfg)
    client = MlflowClient(tracking_uri=tracking_uri)
    model_name = _registered_model_name(task, cfg)
    alias = cfg.mlflow_champion_alias

    mv = client.get_model_version_by_alias(name=model_name, alias=alias)
    # MLflow 3.x returns tags as a plain dict; older versions returned a list of tag objects.
    raw_tags = mv.tags or {}
    tags_dict = raw_tags if isinstance(raw_tags, dict) else {t.key: t.value for t in raw_tags}

    label_names_raw = tags_dict.get("label_names", "")
    label_names = label_names_raw.split(",") if label_names_raw else []

    return {
        "registered_model_name": model_name,
        "version": str(mv.version),  # MLflow 3.x returns int; normalise to str.
        "run_id": mv.run_id,
        "algo_name": tags_dict.get("algo_name", ""),
        "task": task,
        "label_names": label_names,
        "cv_macro_f1": float(tags_dict.get("cv_macro_f1", 0.0)),
        "joblib_path": tags_dict.get("joblib_path", ""),
        "alias": alias,
        "creation_timestamp": mv.creation_timestamp,
        "tags": tags_dict,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_register_and_promote(
    task: str,
    mlflow_uri: str | None = None,
    output_dir: Path | None = None,
    cfg: Config | None = None,
    promote: bool = True,
) -> dict[str, Any]:
    """
    Register the best candidate for *task* and optionally promote to champion.

    Parameters
    ----------
    task:
        ``"health"``, ``"restoration"``, or ``"all"``.
    mlflow_uri:
        SQLite tracking URI override.
    output_dir:
        Directory containing evaluation JSON and joblib files.
    cfg:
        Config instance.
    promote:
        If True and quality gate passes, set the champion alias.

    Returns
    -------
    Dict with one key per task, each mapping to a sub-dict with registration
    and promotion details.
    """
    cfg = cfg or get_config()
    tasks = list(_VALID_TASKS) if task == "all" else [task]
    results: dict[str, Any] = {}

    for t in tasks:
        reg = register_candidate(t, mlflow_uri=mlflow_uri, output_dir=output_dir, cfg=cfg)
        promoted = False
        if promote:
            if reg.gate.passed:
                promote_champion(t, reg.version, mlflow_uri=mlflow_uri, cfg=cfg)
                reg.champion_set = True
                promoted = True
            else:
                logger.warning(
                    "[%s] Quality gate FAILED — champion alias NOT updated. Failures: %s",
                    t,
                    reg.gate.failures,
                )
        results[t] = {
            "registered_model_name": reg.registered_model_name,
            "version": reg.version,
            "run_id": reg.run_id,
            "algo_name": reg.algo_name,
            "cv_macro_f1": reg.cv_macro_f1,
            "gate_passed": reg.gate.passed,
            "gate_failures": reg.gate.failures,
            "champion_set": promoted,
        }
        logger.info(
            "[%s] Registration complete: version=%s gate=%s champion=%s",
            t,
            reg.version,
            "PASS" if reg.gate.passed else "FAIL",
            promoted,
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for model registration and promotion.

    Exit codes
    ----------
    0   Success.
    1   Runtime error.
    2   Invalid arguments.
    """
    import argparse

    setup_logging(__name__)
    parser = argparse.ArgumentParser(
        description=("Register and promote CoralSense models in the MLflow Model Registry."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=["health", "restoration", "all"],
        default="all",
        help="Which task to register/promote.",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=None,
        help="MLflow tracking URI (default: from config/env).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Directory containing evaluation JSON and model files.",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register the best candidate (without promoting).",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Register AND promote the champion alias.",
    )
    args = parser.parse_args(argv)

    if not args.register and not args.promote:
        parser.error("Specify --register or --promote (or both).")

    try:
        cfg = get_config()
        results = run_register_and_promote(
            task=args.task,
            mlflow_uri=args.mlflow_uri,
            output_dir=args.output_dir,
            cfg=cfg,
            promote=args.promote,
        )
        for task, r in results.items():
            gate_str = "PASS" if r["gate_passed"] else f"FAIL ({r['gate_failures']})"
            champ_str = "YES" if r["champion_set"] else "NO"
            print(
                f"[{task}] model={r['registered_model_name']} "
                f"version={r['version']} "
                f"algo={r['algo_name']} "
                f"cv_macro_f1={r['cv_macro_f1']:.4f} "
                f"gate={gate_str} "
                f"champion={champ_str}"
            )
        return 0
    except Exception as exc:
        logger.error("Registration failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
