"""
src/dashboard/views/9_Governance.py — Read-only model governance dashboard page.

Displays:
- Champion model metadata (from MLflow registry via FastAPI /model-info)
- Drift monitoring status and RETRAIN recommendation
- Labelled-data retraining contract
- Promotion and rollback history (from receipt files)
- Governance explanation

Read-only: contains no training controls, promotion controls, or rollback controls.
Does not mutate the MLflow registry or any model artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.dashboard import theme
from src.dashboard.components import render_sidebar, set_page

set_page("Model Governance", icon="🔒")
render_sidebar(show_region_filter=False)

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_REPORTS = _ROOT / "reports"

# ── Helpers ────────────────────────────────────────────────────────────────────


def _load_json(path: Path) -> dict | None:
    """Load a JSON file, returning None on any failure."""
    try:
        if path.exists() and path.stat().st_size > 0:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        pass
    return None


def _find_receipts(prefix: str) -> list[dict]:
    """Find all JSON receipt files matching the given prefix."""
    receipts = []
    for p in sorted(_REPORTS.glob(f"{prefix}_*.json"), reverse=True):
        data = _load_json(p)
        if data:
            data["_filename"] = p.name
            receipts.append(data)
    return receipts


def _champion_info() -> dict | None:
    """Get champion metadata from FastAPI /model-info (graceful fallback)."""
    try:
        import urllib.request

        api_url = _get_api_url()
        url = f"{api_url}/model-info"
        with urllib.request.urlopen(url, timeout=4) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _get_api_url() -> str:
    import os

    return os.environ.get("CORALSENSE_API_URL", "http://127.0.0.1:8000")


# ── Page layout ────────────────────────────────────────────────────────────────
theme.page_header(
    "Model Governance",
    "Read-only view. No training, promotion, or rollback controls are available here.",
    eyebrow="Accountability",
)
st.warning(
    "**Synthetic-data disclaimer:** All models were trained on a **synthetic prototype "
    "dataset** whose labels are **algorithmically generated** from variables that are "
    "themselves supplied to the models as predictors (**label-construction leakage / "
    "circular supervision**). Metrics demonstrate recovery of synthetic structure and "
    "MLOps pipeline behaviour, **not** real-world coral reef prediction accuracy. No "
    "independent biological ground truth exists in this project and no external "
    "validation has been performed.",
    icon="⚠️",
)

# ── Section 1: Champion models ─────────────────────────────────────────────────
theme.section("Champion Models", kicker="Registry")

champion_data = _champion_info()
if champion_data:
    models_info = champion_data.get("models", {})
    cols = st.columns(2)
    for i, (task, meta) in enumerate(models_info.items()):
        with cols[i % 2]:
            task_label = task.replace("_", " ").title()
            model_name = meta.get("model_name", task)
            algorithm = meta.get("algorithm", "unknown")
            version = meta.get("version", "?")
            alias = meta.get("alias", "champion")
            st.metric(label=f"{task_label} — champion alias", value=f"v{version}")
            st.markdown(f"**Model name:** `{model_name}`")
            st.markdown(f"**Algorithm:** `{algorithm}`")
            st.markdown(f"**Alias:** `{alias}`")
else:
    st.info(
        "FastAPI service is not reachable — champion metadata not available. "
        "Start the API with `make api` or `python scripts/demo.py start`.",
        icon="ℹ️",
    )

# ── Section 2: Drift status ────────────────────────────────────────────────────
theme.section("Drift Monitoring Status", kicker="Model health")

drift_path = _REPORTS / "drift_summary.json"
drift = _load_json(drift_path)

if drift:
    recommendation = drift.get("recommendation", "UNKNOWN")
    color = "red" if recommendation == "RETRAIN" else "green"
    st.markdown(f"**Recommendation:** :{color}[**{recommendation}**]")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_drift = drift.get("n_drifted_features", 0)
        st.metric("Drifted features", n_drift)
    with col2:
        pred_drift = drift.get("prediction_drift", {})
        health_drift = pred_drift.get("health", False)
        st.metric("Health prediction drift", "YES" if health_drift else "NO")
    with col3:
        rest_drift = pred_drift.get("restoration", False)
        st.metric("Restoration prediction drift", "YES" if rest_drift else "NO")

    with st.expander("Full drift summary"):
        st.json(drift)
else:
    st.info(
        "No drift summary found. Generate one with `make drift` or "
        "`python -m src.monitoring.run_drift --no-html`.",
        icon="ℹ️",
    )

# ── Section 3: Retraining data contract ───────────────────────────────────────
theme.section("Retraining Data Contract", kicker="Guardrails")

st.markdown(
    """
The retraining pipeline (`scripts/run_retraining.py`) enforces the following
checks before any challenger model can be trained.
**The unlabelled drift production window is rejected at check 1.**

| # | Requirement | Detail |
|---|---|---|
| 1 | Both target columns present | `reef_health` and/or `restoration_suitability` must be non-null |
| 2 | No NaN targets | Rows with missing labels are rejected |
| 3 | Minimum row count | At least 200 labelled rows required (configurable) |
| 4 | All classes present | All label classes must appear in the input |
| 5 | Minimum class count | Each class must have at least 5 examples |
| 6 | Valid feature columns | Input must contain expected sensor feature columns |
| 7 | SHA-256 provenance | Input file hash recorded in MLflow tags |
| 8 | Retraining permission | Drift RETRAIN recommendation OR manual `--reason` supplied |

Champion promotion additionally requires:
- A completed comparison report (`reports/comparison_*.json`)
- `--approve`, `--approver`, and `--reason` CLI flags
- Quality gate re-validation (challenger must meet minimum thresholds)
"""
)

# ── Section 4: Comparison reports ─────────────────────────────────────────────
theme.section("Latest Challenger Comparison", kicker="Evaluation")

comparison_files = sorted(_REPORTS.glob("comparison_*.json"), reverse=True)
if comparison_files:
    latest = _load_json(comparison_files[0])
    if latest:
        outcome = latest.get("outcome", "unknown")
        outcome_color = {
            "eligible_for_promotion": "green",
            "review_required": "orange",
            "reject": "red",
        }.get(outcome, "gray")
        st.markdown(f"**Outcome:** :{outcome_color}[**{outcome}**]")
        st.markdown(f"**File:** `{comparison_files[0].name}`")

        gate_results = latest.get("gate_results", {})
        if gate_results:
            rows = []
            for gate_name, gate_data in gate_results.items():
                if isinstance(gate_data, dict):
                    rows.append(
                        {
                            "Gate": gate_name,
                            "Passed": "YES" if gate_data.get("passed") else "NO",
                            "Detail": str(gate_data.get("detail", "")),
                        }
                    )
            if rows:
                import pandas as pd

                st.dataframe(pd.DataFrame(rows), use_container_width=True)

        with st.expander("Full comparison report"):
            st.json(latest)
else:
    st.info(
        "No comparison reports found. Run a retraining dry-run to see one: "
        "`python scripts/run_retraining.py --task health --input data/raw/observations.csv "
        "--drift-summary reports/drift_summary.json --dry-run`",
        icon="ℹ️",
    )

# ── Section 5: Promotion history ──────────────────────────────────────────────
theme.section("Promotion History", kicker="Audit trail")

promotion_receipts = _find_receipts("promotion_receipt")
if promotion_receipts:
    for receipt in promotion_receipts:
        with st.expander(receipt.get("_filename", "receipt")):
            st.json({k: v for k, v in receipt.items() if k != "_filename"})
else:
    st.info("No promotion receipts found (champion has not been changed from v1).", icon="ℹ️")

# ── Section 6: Rollback history ───────────────────────────────────────────────
theme.section("Rollback History", kicker="Audit trail")

rollback_receipts = _find_receipts("rollback_receipt")
if rollback_receipts:
    for receipt in rollback_receipts:
        with st.expander(receipt.get("_filename", "receipt")):
            st.json({k: v for k, v in receipt.items() if k != "_filename"})
else:
    st.info("No rollback receipts found.", icon="ℹ️")

# ── Section 7: Governance explanation ─────────────────────────────────────────
theme.section("Governance Explanation", kicker="Policy")

st.markdown(
    """
### Why explicit approval is required

The champion alias in the MLflow Model Registry can only be moved by running
`src/models/promote.py` with all three mandatory flags:

```bash
python -m src.models.promote \\
  --model coralsense_reef_health \\
  --challenger-version <N> \\
  --approve \\
  --approver "Your Name" \\
  --reason "Challenger improves macro-F1 by X pp over champion"
```

This design prevents silent model regression: no pipeline, container, or
background process can silently replace the production model.

### Rollback

Rolling back restores a previous champion version without deleting any
model version:

```bash
python -m src.models.rollback \\
  --model coralsense_reef_health \\
  --target-version 1 \\
  --approver "Your Name" \\
  --reason "Challenger underperforms on held-out data"
```

A rollback receipt is always written to `reports/rollback_receipt_<timestamp>.json`.

### Synthetic-data limitation

All model versions currently registered in this project were trained on
the synthetic prototype dataset. Metrics do not generalise to real coral
reef environments.

### Label-construction leakage / circular supervision

The limitation is stronger than "the data is synthetic". Both targets are
**algorithmically generated**: `src/data/generate_data.py` computes a weighted
score over other columns in the same file, adds Gaussian noise, and thresholds
the result. Those same generating columns are then supplied to the models as
predictors.

Each target column is correctly excluded from the other task's feature matrix,
so this is **not** ordinary direct target leakage — it is
**label-construction leakage**. Three engineered features
(`thermal_stress_index`, `oxygen_stress_index`, `water_quality_index`)
additionally reproduce components of the health-score formula exactly, which is
**engineered proxy leakage**.

**Audit diagnostic.** A closed-form re-computation of the generator's formula,
with its noise term removed and no machine learning at all, reproduces the
stored labels at macro-F1 ≈ 0.770 (health) and ≈ 0.812 (restoration) —
matching or exceeding an equivalent trained model (≈ 0.760 / ≈ 0.791). These are
**audit diagnostics**, not registered model metrics, and they do not replace the
champion figures shown above.

Consequence: registered metrics are evidence of a correctly functioning ML and
MLOps pipeline and are valid for relative algorithm comparison. They are **not**
evidence of ecological validity. Full analysis:
`docs/audits/dataset_scientific_audit_2026-08-19.md`.

### Before any real-world deployment

1. Replace the synthetic generator with a real sensor ingestion module.
2. Commission ecological surveys to produce genuinely labelled ground truth.
3. Re-specify both targets so the label is not constructed from variables that
   are also supplied as predictors.
4. Validate against an independent real dataset — no internal split can validate
   a model whose labels were authored by the same script as its features.
5. Obtain domain expert sign-off before using predictions to guide any
   conservation management decision.
"""
)

st.divider()
st.caption("Oceanographic MLOps — Governance page — read-only — no model mutations occur here")
