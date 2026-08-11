"""
tests/test_dashboard.py — Test suite for the CoralSense Streamlit dashboard (M10).

Test strategy
-------------
- API client: mock requests.get / requests.post — no real network calls.
- Data loader: use temporary CSV / JSON files — no canonical dataset modified.
- Components: test colour maps, disclaimer text, payload builder.
- Importability: verify every page module can be imported without side effects.
- AppTest smoke tests: run selected pages with mocked data_loader calls.
- Safety: assert no model-training, registry, or MLflow imports in dashboard code.
- No graphical display required (headless).

What is intentionally NOT tested here
--------------------------------------
- Real API calls to a running FastAPI server (use integration tests for that).
- Actual Streamlit rendering quality (visual appearance).
- End-to-end browser interaction.
"""

from __future__ import annotations

import ast
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any
from unittest import mock

import pandas as pd
import pytest
import requests

# ---------------------------------------------------------------------------
# Project root & sample data helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DASHBOARD_DIR = _PROJECT_ROOT / "src" / "dashboard"
_PAGES_DIR = _DASHBOARD_DIR / "pages"


def resolve_app_path(script_path: str | Path) -> Path:
    """
    Return an absolute, existing path to a dashboard script.

    ``AppTest.from_file`` documents its ``script_path`` as "absolute or
    relative to the file calling ``.from_file``".  A relative path such as
    ``"src/dashboard/app.py"`` is therefore resolved against ``tests/`` and
    becomes ``tests/src/dashboard/app.py``.  Some Streamlit versions probe the
    path against the current working directory first, which accidentally works
    when pytest is launched from the repository root — but that is not a
    guarantee, and it breaks in CI.

    Every path handed to ``AppTest`` is made absolute here, resolved against
    the repository root (derived from this file) and never against the cwd.
    """
    path = Path(script_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise AssertionError(
            f"dashboard script not found: {path} (requested {script_path!r}, "
            f"project root {_PROJECT_ROOT})"
        )
    return path


HEALTH_CLASSES = ("healthy", "stressed", "bleached", "severely_degraded")
RESTORATION_CLASSES = ("suitable", "moderately_suitable", "unsuitable")
REGIONS = (
    "Lakshadweep",
    "Gulf of Mannar",
    "Gulf of Kutch",
    "Andaman and Nicobar Islands",
)


def _make_sample_df(n: int = 50) -> pd.DataFrame:
    """Return a minimal DataFrame matching the expected schema."""
    import numpy as np

    rng = np.random.default_rng(42)
    health_labels = list(HEALTH_CLASSES)
    rest_labels = list(RESTORATION_CLASSES)

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "latitude": rng.uniform(8.0, 14.0, n),
            "longitude": rng.uniform(72.0, 94.0, n),
            "region": [REGIONS[i % 4] for i in range(n)],
            "depth_m": rng.uniform(0, 50, n),
            "water_temperature_c": rng.uniform(22, 32, n),
            "ph": rng.uniform(7.8, 8.4, n),
            "salinity_ppt": rng.uniform(28, 38, n),
            "dissolved_oxygen_mg_l": rng.uniform(5, 10, n),
            "turbidity_ntu": rng.uniform(0, 20, n),
            "light_intensity": rng.uniform(100, 2000, n),
            "current_speed_m_s": rng.uniform(0, 1, n),
            "sonar_backscatter": rng.uniform(-50, -5, n),
            "rugosity_index": rng.uniform(1, 8, n),
            "hard_substrate_percentage": rng.uniform(10, 90, n),
            "acoustic_complexity_index": rng.uniform(0.1, 0.9, n),
            "coral_cover_percentage": rng.uniform(5, 80, n),
            "bleaching_percentage": rng.uniform(0, 30, n),
            "disease_percentage": rng.uniform(0, 15, n),
            "reef_health": [health_labels[i % 4] for i in range(n)],
            "restoration_suitability": [rest_labels[i % 3] for i in range(n)],
        }
    )


def _make_eval_json(task: str) -> dict[str, Any]:
    if task == "health":
        label_names = list(HEALTH_CLASSES)
        best = "logistic_regression"
    else:
        label_names = list(RESTORATION_CLASSES)
        best = "xgboost"

    def _model_entry():
        return {
            "cv_macro_f1_mean": 0.75,
            "cv_macro_f1_std": 0.01,
            "cv_balanced_accuracy_mean": 0.77,
            "cv_balanced_accuracy_std": 0.01,
            "test_accuracy": 0.80,
            "test_balanced_accuracy": 0.78,
            "test_macro_precision": 0.74,
            "test_macro_recall": 0.76,
            "test_macro_f1": 0.75,
            "test_weighted_f1": 0.79,
            "test_per_class": {
                cls: {"precision": 0.75, "recall": 0.75, "f1": 0.75, "support": 100}
                for cls in label_names
            },
        }

    return {
        "task": task,
        "best_model_name": best,
        "best_cv_macro_f1": 0.75,
        "label_names": label_names,
        "training_duration_s": 5.0,
        "models": {
            "logistic_regression": _model_entry(),
            "random_forest": _model_entry(),
            "xgboost": _model_entry(),
        },
    }


def _make_predict_response() -> dict[str, Any]:
    return {
        "health": {
            "predicted_class": "healthy",
            "probabilities": {
                "healthy": 0.80,
                "stressed": 0.15,
                "bleached": 0.03,
                "severely_degraded": 0.02,
            },
            "confidence": 0.80,
            "task": "health",
            "registered_model_name": "coralsense_reef_health",
            "model_version": "1",
            "model_alias": "champion",
            "run_id": "abc123",
            "prediction_timestamp": "2026-07-14T10:00:00+00:00",
            "synthetic_data_disclaimer": "Synthetic data only.",
        },
        "restoration": {
            "predicted_class": "suitable",
            "probabilities": {
                "suitable": 0.90,
                "moderately_suitable": 0.08,
                "unsuitable": 0.02,
            },
            "confidence": 0.90,
            "task": "restoration",
            "registered_model_name": "coralsense_restoration_suitability",
            "model_version": "1",
            "model_alias": "champion",
            "run_id": "def456",
            "prediction_timestamp": "2026-07-14T10:00:00+00:00",
            "synthetic_data_disclaimer": "Synthetic data only.",
        },
    }


# ---------------------------------------------------------------------------
# API client tests
# ---------------------------------------------------------------------------


class TestAPIClientHealth:
    """Tests for APIClient.health()."""

    def test_health_success(self):
        from src.dashboard.api_client import APIClient

        mock_response = mock.Mock()
        mock_response.json.return_value = {
            "status": "ok",
            "health_model_ready": True,
            "restoration_model_ready": True,
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("requests.get", return_value=mock_response) as m:
            client = APIClient(base_url="http://test:8000")
            result = client.health()
            m.assert_called_once_with("http://test:8000/health", timeout=10)
            assert result["status"] == "ok"

    def test_health_connection_error(self):
        from src.dashboard.api_client import APIClient, APIError

        with mock.patch("requests.get", side_effect=requests.ConnectionError("refused")):
            client = APIClient(base_url="http://test:8000")
            with pytest.raises(APIError, match="Cannot connect"):
                client.health()

    def test_health_timeout(self):
        from src.dashboard.api_client import APIClient, APIError

        with mock.patch("requests.get", side_effect=requests.Timeout()):
            client = APIClient(base_url="http://test:8000", timeout=1)
            with pytest.raises(APIError, match="timed out"):
                client.health()

    def test_health_http_error_503(self):
        from src.dashboard.api_client import APIClient, APIError

        mock_resp = mock.Mock()
        mock_resp.status_code = 503
        http_err = requests.HTTPError(response=mock_resp)
        mock_resp.raise_for_status.side_effect = http_err

        with mock.patch("requests.get", return_value=mock_resp):
            client = APIClient(base_url="http://test:8000")
            with pytest.raises(APIError, match="HTTP 503"):
                client.health()

    def test_health_malformed_response_not_dict(self):
        from src.dashboard.api_client import APIClient, APIError

        mock_resp = mock.Mock()
        mock_resp.json.return_value = ["not", "a", "dict"]
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch("requests.get", return_value=mock_resp):
            client = APIClient(base_url="http://test:8000")
            with pytest.raises(APIError, match="Unexpected response format"):
                client.health()


class TestAPIClientModelInfo:
    """Tests for APIClient.model_info()."""

    def test_model_info_success(self):
        from src.dashboard.api_client import APIClient

        mock_resp = mock.Mock()
        mock_resp.json.return_value = {
            "health": {"available": True},
            "restoration": {"available": True},
            "synthetic_data_disclaimer": "Synthetic only.",
        }
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch("requests.get", return_value=mock_resp):
            client = APIClient(base_url="http://test:8000")
            result = client.model_info()
            assert result["health"]["available"] is True

    def test_model_info_connection_error(self):
        from src.dashboard.api_client import APIClient, APIError

        with mock.patch("requests.get", side_effect=requests.ConnectionError()):
            with pytest.raises(APIError):
                APIClient(base_url="http://test:8000").model_info()


class TestAPIClientPredictBoth:
    """Tests for APIClient.predict_both()."""

    def test_predict_both_success(self):
        from src.dashboard.api_client import APIClient

        expected = _make_predict_response()
        mock_resp = mock.Mock()
        mock_resp.json.return_value = expected
        mock_resp.raise_for_status = mock.Mock()

        payload = {"region": "Gulf of Mannar", "depth_m": 5.0}
        with mock.patch("requests.post", return_value=mock_resp) as m:
            client = APIClient(base_url="http://test:8000")
            result = client.predict_both(payload)
            m.assert_called_once_with("http://test:8000/predict/both", json=payload, timeout=10)
            assert result["health"]["predicted_class"] == "healthy"
            assert result["restoration"]["predicted_class"] == "suitable"

    def test_predict_both_timeout(self):
        from src.dashboard.api_client import APIClient, APIError

        with mock.patch("requests.post", side_effect=requests.Timeout()):
            with pytest.raises(APIError, match="timed out"):
                APIClient(base_url="http://test:8000").predict_both({})

    def test_predict_both_connection_error(self):
        from src.dashboard.api_client import APIClient, APIError

        with mock.patch("requests.post", side_effect=requests.ConnectionError()):
            with pytest.raises(APIError, match="Cannot connect"):
                APIClient(base_url="http://test:8000").predict_both({})

    def test_predict_both_http_error(self):
        from src.dashboard.api_client import APIClient, APIError

        mock_resp = mock.Mock()
        mock_resp.status_code = 422
        http_err = requests.HTTPError(response=mock_resp)
        mock_resp.raise_for_status.side_effect = http_err

        with mock.patch("requests.post", return_value=mock_resp):
            with pytest.raises(APIError, match="HTTP 422"):
                APIClient(base_url="http://test:8000").predict_both({})

    def test_predict_both_malformed_response(self):
        from src.dashboard.api_client import APIClient, APIError

        mock_resp = mock.Mock()
        mock_resp.json.return_value = "not a dict"
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch("requests.post", return_value=mock_resp):
            with pytest.raises(APIError, match="Unexpected response format"):
                APIClient(base_url="http://test:8000").predict_both({})


class TestAPIClientEnvConfig:
    """Tests for environment-variable-based configuration."""

    def test_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("CORALSENSE_API_URL", "http://custom:9999")
        from src.dashboard import api_client

        importlib.reload(api_client)
        client = api_client.APIClient()
        assert client.base_url == "http://custom:9999"

    def test_default_base_url(self):
        from src.dashboard.api_client import APIClient

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CORALSENSE_API_URL", None)
            client = APIClient()
            assert "127.0.0.1" in client.base_url or "localhost" in client.base_url

    def test_timeout_from_env(self, monkeypatch):
        monkeypatch.setenv("CORALSENSE_API_TIMEOUT", "30")
        from src.dashboard import api_client

        importlib.reload(api_client)
        client = api_client.APIClient()
        assert client.timeout == 30


# ---------------------------------------------------------------------------
# Data loader tests
# ---------------------------------------------------------------------------


class TestLoadObservations:
    """Tests for data_loader.load_observations()."""

    def test_load_observations_success(self, tmp_path, monkeypatch):
        from src.dashboard import data_loader

        csv_path = tmp_path / "observations.csv"
        df_orig = _make_sample_df(30)
        df_orig.to_csv(csv_path, index=False)

        monkeypatch.setattr(data_loader, "_RAW_CSV", csv_path)
        data_loader.load_observations.clear()  # clear Streamlit cache

        df = data_loader.load_observations()
        assert len(df) == 30
        assert "reef_health" in df.columns
        assert "restoration_suitability" in df.columns

    def test_load_observations_missing_file(self, tmp_path, monkeypatch):
        from src.dashboard import data_loader

        monkeypatch.setattr(data_loader, "_RAW_CSV", tmp_path / "nonexistent.csv")
        data_loader.load_observations.clear()

        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            data_loader.load_observations()

    def test_load_observations_missing_columns(self, tmp_path, monkeypatch):
        from src.dashboard import data_loader

        csv_path = tmp_path / "bad.csv"
        pd.DataFrame({"only_one_col": [1, 2, 3]}).to_csv(csv_path, index=False)

        monkeypatch.setattr(data_loader, "_RAW_CSV", csv_path)
        data_loader.load_observations.clear()

        with pytest.raises(ValueError, match="missing required columns"):
            data_loader.load_observations()


class TestLoadEvaluation:
    """Tests for data_loader.load_evaluation()."""

    def test_load_evaluation_health_success(self, tmp_path, monkeypatch):
        from src.dashboard import data_loader

        monkeypatch.setattr(data_loader, "_EVAL_DIR", tmp_path)
        data_loader.load_evaluation.clear()

        eval_data = _make_eval_json("health")
        (tmp_path / "evaluation_health.json").write_text(json.dumps(eval_data))

        result = data_loader.load_evaluation("health")
        assert result is not None
        assert result["task"] == "health"
        assert "models" in result

    def test_load_evaluation_restoration_success(self, tmp_path, monkeypatch):
        from src.dashboard import data_loader

        monkeypatch.setattr(data_loader, "_EVAL_DIR", tmp_path)
        data_loader.load_evaluation.clear()

        eval_data = _make_eval_json("restoration")
        (tmp_path / "evaluation_restoration.json").write_text(json.dumps(eval_data))

        result = data_loader.load_evaluation("restoration")
        assert result is not None
        assert result["best_model_name"] == "xgboost"

    def test_load_evaluation_missing_file_returns_none(self, tmp_path, monkeypatch):
        from src.dashboard import data_loader

        monkeypatch.setattr(data_loader, "_EVAL_DIR", tmp_path)
        data_loader.load_evaluation.clear()

        result = data_loader.load_evaluation("health")
        assert result is None


class TestSampleForDisplay:
    """Tests for data_loader.sample_for_display()."""

    def test_sample_bounded_to_max_n(self):
        from src.dashboard.data_loader import sample_for_display

        df = _make_sample_df(500)
        sampled = sample_for_display(df, max_n=100)
        assert len(sampled) == 100

    def test_sample_small_df_returns_copy(self):
        from src.dashboard.data_loader import sample_for_display

        df = _make_sample_df(20)
        sampled = sample_for_display(df, max_n=100)
        assert len(sampled) == 20

    def test_sample_does_not_mutate_original(self):
        from src.dashboard.data_loader import sample_for_display

        df = _make_sample_df(200)
        original_len = len(df)
        original_hash = hash(df["reef_health"].iloc[0])
        _ = sample_for_display(df, max_n=50)
        assert len(df) == original_len
        assert hash(df["reef_health"].iloc[0]) == original_hash

    def test_sample_is_reproducible(self):
        from src.dashboard.data_loader import sample_for_display

        df = _make_sample_df(500)
        s1 = sample_for_display(df, max_n=100, seed=42)
        s2 = sample_for_display(df, max_n=100, seed=42)
        assert list(s1.index) == list(s2.index)


# ---------------------------------------------------------------------------
# Components tests
# ---------------------------------------------------------------------------


class TestColorMaps:
    """Tests for colour map completeness."""

    def test_health_colors_cover_all_classes(self):
        from src.dashboard.components import HEALTH_COLORS

        for cls in HEALTH_CLASSES:
            assert cls in HEALTH_COLORS, f"Missing health color for '{cls}'"

    def test_restoration_colors_cover_all_classes(self):
        from src.dashboard.components import RESTORATION_COLORS

        for cls in RESTORATION_CLASSES:
            assert cls in RESTORATION_COLORS, f"Missing restoration color for '{cls}'"

    def test_health_colors_are_hex(self):
        from src.dashboard.components import HEALTH_COLORS

        for cls, color in HEALTH_COLORS.items():
            assert color.startswith("#"), f"Health color for '{cls}' is not a hex string"

    def test_restoration_colors_are_hex(self):
        from src.dashboard.components import RESTORATION_COLORS

        for cls, color in RESTORATION_COLORS.items():
            assert color.startswith("#"), f"Restoration color for '{cls}' is not hex"

    def test_healthy_class_is_green(self):
        from src.dashboard.components import HEALTH_COLORS

        # Green hue: R < G, R < B roughly; simpler: check green starts #2e or similar
        green = HEALTH_COLORS["healthy"]
        r, g, b = int(green[1:3], 16), int(green[3:5], 16), int(green[5:7], 16)
        assert g > r and g > b, "healthy class should be green-dominant"

    def test_unsuitable_class_is_red(self):
        from src.dashboard.components import RESTORATION_COLORS

        red = RESTORATION_COLORS["unsuitable"]
        r, g, b = int(red[1:3], 16), int(red[3:5], 16), int(red[5:7], 16)
        assert r > g and r > b, "unsuitable class should be red-dominant"


class TestDisclaimer:
    """Tests for disclaimer text content."""

    def test_disclaimer_short_contains_synthetic(self):
        from src.dashboard.components import DISCLAIMER_SHORT

        assert "synthetic" in DISCLAIMER_SHORT.lower()

    def test_disclaimer_full_contains_no_real_data(self):
        from src.dashboard.components import DISCLAIMER_FULL

        text = DISCLAIMER_FULL.lower()
        assert "synthetic" in text
        assert "not" in text

    def test_disclaimer_short_present(self):
        from src.dashboard.components import DISCLAIMER_SHORT

        assert len(DISCLAIMER_SHORT) > 20


class TestBuildPredictPayload:
    """Tests for build_predict_payload() target-label exclusion."""

    def test_excludes_reef_health(self):
        from src.dashboard.components import build_predict_payload

        form = {"region": "Gulf of Mannar", "depth_m": 5.0, "reef_health": "healthy"}
        payload = build_predict_payload(form)
        assert "reef_health" not in payload

    def test_excludes_restoration_suitability(self):
        from src.dashboard.components import build_predict_payload

        form = {
            "region": "Gulf of Mannar",
            "depth_m": 5.0,
            "restoration_suitability": "suitable",
        }
        payload = build_predict_payload(form)
        assert "restoration_suitability" not in payload

    def test_retains_all_inference_features(self):
        from src.dashboard.components import build_predict_payload

        form = {
            "region": "Gulf of Mannar",
            "depth_m": 5.0,
            "water_temperature_c": 27.5,
            "reef_health": "healthy",
            "restoration_suitability": "suitable",
        }
        payload = build_predict_payload(form)
        assert payload["region"] == "Gulf of Mannar"
        assert payload["depth_m"] == 5.0
        assert payload["water_temperature_c"] == 27.5
        assert len(payload) == 3  # 3 inference features, no targets

    def test_empty_form_returns_empty_payload(self):
        from src.dashboard.components import build_predict_payload

        assert build_predict_payload({}) == {}


class TestProbabilityIntegrity:
    """Tests for probability ordering and totals in prediction responses."""

    def test_probabilities_sum_to_one(self):
        response = _make_predict_response()
        for task in ("health", "restoration"):
            probs = response[task]["probabilities"]
            total = sum(probs.values())
            assert math.isclose(total, 1.0, abs_tol=0.01), (
                f"{task} probabilities sum to {total}, expected ~1.0"
            )

    def test_predicted_class_has_highest_probability(self):
        response = _make_predict_response()
        for task in ("health", "restoration"):
            res = response[task]
            best_cls = max(res["probabilities"], key=res["probabilities"].get)
            assert best_cls == res["predicted_class"]

    def test_confidence_matches_max_probability(self):
        response = _make_predict_response()
        for task in ("health", "restoration"):
            res = response[task]
            max_prob = max(res["probabilities"].values())
            assert math.isclose(res["confidence"], max_prob, abs_tol=0.001)

    def test_health_probabilities_cover_all_classes(self):
        response = _make_predict_response()
        probs = response["health"]["probabilities"]
        for cls in HEALTH_CLASSES:
            assert cls in probs, f"Health probability missing for '{cls}'"

    def test_restoration_probabilities_cover_all_classes(self):
        response = _make_predict_response()
        probs = response["restoration"]["probabilities"]
        for cls in RESTORATION_CLASSES:
            assert cls in probs, f"Restoration probability missing for '{cls}'"


# ---------------------------------------------------------------------------
# Filter logic tests
# ---------------------------------------------------------------------------


class TestFilterLogic:
    """Tests for region filtering and data subsetting."""

    def test_region_filter_reduces_rows(self):
        df = _make_sample_df(100)
        filtered = df[df["region"].isin(["Gulf of Mannar"])]
        assert len(filtered) < len(df)
        assert all(filtered["region"] == "Gulf of Mannar")

    def test_all_regions_filter_keeps_all_rows(self):
        df = _make_sample_df(100)
        filtered = df[df["region"].isin(list(REGIONS))]
        assert len(filtered) == len(df)

    def test_empty_region_filter_gives_empty_df(self):
        df = _make_sample_df(100)
        filtered = df[df["region"].isin([])]
        assert len(filtered) == 0

    def test_health_class_filter(self):
        df = _make_sample_df(100)
        filtered = df[df["reef_health"].isin(["healthy"])]
        assert all(filtered["reef_health"] == "healthy")


# ---------------------------------------------------------------------------
# Model performance JSON loading
# ---------------------------------------------------------------------------


class TestModelPerformanceLoading:
    """Tests for evaluation JSON loading and structure."""

    def test_all_model_keys_present(self):
        eval_data = _make_eval_json("health")
        for algo in ("logistic_regression", "random_forest", "xgboost"):
            assert algo in eval_data["models"]

    def test_per_class_keys_present(self):
        eval_data = _make_eval_json("health")
        champion = eval_data["models"]["logistic_regression"]
        per_class = champion["test_per_class"]
        for cls in HEALTH_CLASSES:
            assert cls in per_class

    def test_metric_values_in_range(self):
        eval_data = _make_eval_json("restoration")
        for algo, metrics in eval_data["models"].items():
            for metric in (
                "cv_macro_f1_mean",
                "test_macro_f1",
                "test_accuracy",
            ):
                val = metrics[metric]
                assert 0.0 <= val <= 1.0, f"{algo}.{metric} out of range: {val}"

    def test_absent_eval_file_returns_none(self, tmp_path, monkeypatch):
        from src.dashboard import data_loader

        monkeypatch.setattr(data_loader, "_EVAL_DIR", tmp_path)
        data_loader.load_evaluation.clear()
        result = data_loader.load_evaluation("health")
        assert result is None


# ---------------------------------------------------------------------------
# Form payload construction
# ---------------------------------------------------------------------------


class TestFormPayloadConstruction:
    """Tests that form values are correctly assembled into API payloads."""

    def _make_full_form(self) -> dict:
        return {
            "region": "Lakshadweep",
            "depth_m": 10.0,
            "water_temperature_c": 28.0,
            "ph": 8.1,
            "salinity_ppt": 35.0,
            "dissolved_oxygen_mg_l": 6.5,
            "turbidity_ntu": 3.0,
            "light_intensity": 600.0,
            "current_speed_m_s": 0.3,
            "sonar_backscatter": -20.0,
            "rugosity_index": 3.0,
            "hard_substrate_percentage": 55.0,
            "acoustic_complexity_index": 0.65,
            "coral_cover_percentage": 40.0,
            "bleaching_percentage": 8.0,
            "disease_percentage": 3.0,
            # Target labels that must be excluded
            "reef_health": "stressed",
            "restoration_suitability": "moderately_suitable",
        }

    def test_target_labels_excluded_from_payload(self):
        from src.dashboard.components import build_predict_payload

        payload = build_predict_payload(self._make_full_form())
        assert "reef_health" not in payload
        assert "restoration_suitability" not in payload

    def test_all_16_inference_features_present(self):
        from src.dashboard.components import build_predict_payload
        from src.dashboard.data_loader import INFERENCE_FEATURES

        payload = build_predict_payload(self._make_full_form())
        for feat in INFERENCE_FEATURES:
            assert feat in payload, f"Missing inference feature: {feat}"

    def test_payload_numeric_values_are_correct_type(self):
        from src.dashboard.components import build_predict_payload

        payload = build_predict_payload(self._make_full_form())
        assert isinstance(payload["depth_m"], float)
        assert isinstance(payload["water_temperature_c"], float)

    def test_payload_region_is_string(self):
        from src.dashboard.components import build_predict_payload

        payload = build_predict_payload(self._make_full_form())
        assert isinstance(payload["region"], str)


# ---------------------------------------------------------------------------
# Safety checks — no forbidden imports in dashboard code
# ---------------------------------------------------------------------------

_DASHBOARD_FILES = list((_PROJECT_ROOT / "src" / "dashboard").rglob("*.py"))


class TestNoForbiddenImports:
    """Assert that dashboard modules never import model-training or registry code."""

    def _collect_imports(self, path: Path) -> list[str]:
        """Return all top-level import module names in a Python source file."""
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            return []
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports

    @pytest.mark.parametrize("path", _DASHBOARD_FILES, ids=lambda p: p.name)
    def test_no_model_training_import(self, path):
        imports = self._collect_imports(path)
        for imp in imports:
            assert "src.models.train" not in imp, f"{path.name} imports model training code: {imp}"

    @pytest.mark.parametrize("path", _DASHBOARD_FILES, ids=lambda p: p.name)
    def test_no_registry_mutation_import(self, path):
        imports = self._collect_imports(path)
        for imp in imports:
            assert "src.models.registry" not in imp, f"{path.name} imports registry code: {imp}"

    @pytest.mark.parametrize("path", _DASHBOARD_FILES, ids=lambda p: p.name)
    def test_no_mlflow_direct_import(self, path):
        imports = self._collect_imports(path)
        for imp in imports:
            assert imp != "mlflow", f"{path.name} directly imports mlflow: {imp}"

    @pytest.mark.parametrize("path", _DASHBOARD_FILES, ids=lambda p: p.name)
    def test_no_hardcoded_home_paths(self, path):
        content = path.read_text()
        assert "/home/" not in content, f"{path.name} contains a hardcoded /home/ path"

    @pytest.mark.parametrize("path", _DASHBOARD_FILES, ids=lambda p: p.name)
    def test_no_canonical_mlflow_db_access(self, path):
        content = path.read_text()
        assert "mlruns.db" not in content, f"{path.name} references the canonical MLflow database"


# ---------------------------------------------------------------------------
# Disclaimer presence in page files
# ---------------------------------------------------------------------------

_PAGE_FILES = list((_PROJECT_ROOT / "src" / "dashboard" / "pages").rglob("*.py"))


class TestDisclaimerPresence:
    """Every page should reference synthetic data or disclaimers."""

    @pytest.mark.parametrize("path", _PAGE_FILES, ids=lambda p: p.name)
    def test_page_references_synthetic(self, path):
        content = path.read_text().lower()
        assert "synthetic" in content or "disclaimer" in content, (
            f"{path.name} does not reference synthetic data or disclaimer"
        )


# ---------------------------------------------------------------------------
# Importability tests
# ---------------------------------------------------------------------------


class TestModuleImportability:
    """Dashboard utility modules must be importable without side effects."""

    def test_api_client_importable(self):
        from src.dashboard import api_client  # noqa: F401

        assert hasattr(api_client, "APIClient")
        assert hasattr(api_client, "APIError")
        assert hasattr(api_client, "get_client")

    def test_data_loader_importable(self):
        from src.dashboard import data_loader  # noqa: F401

        assert hasattr(data_loader, "load_observations")
        assert hasattr(data_loader, "load_evaluation")
        assert hasattr(data_loader, "sample_for_display")

    def test_components_importable(self):
        from src.dashboard import components  # noqa: F401

        assert hasattr(components, "HEALTH_COLORS")
        assert hasattr(components, "RESTORATION_COLORS")
        assert hasattr(components, "DISCLAIMER_FULL")
        assert hasattr(components, "build_predict_payload")
        assert hasattr(components, "render_prediction")

    def test_init_importable(self):
        from src.dashboard import __init__  # noqa: F401


# ---------------------------------------------------------------------------
# AppTest smoke tests (selected pages with mocked dependencies)
# ---------------------------------------------------------------------------


class TestAppSmoke:
    """Smoke tests using Streamlit's AppTest.from_file()."""

    def _get_at(self, page_path: str | Path):
        from streamlit.testing.v1 import AppTest

        # Always absolute: a relative path would be resolved against tests/.
        return AppTest.from_file(str(resolve_app_path(page_path)), default_timeout=30)

    def test_main_app_runs_without_exception(self, tmp_path, monkeypatch):
        from src.dashboard import data_loader

        monkeypatch.setattr(data_loader, "_RAW_CSV", tmp_path / "none.csv")
        data_loader.load_observations.clear()

        # Main app gracefully handles missing dataset — no st.exception() rendered
        with mock.patch("src.dashboard.api_client.APIClient") as MockClient:
            MockClient.return_value.health.return_value = {"status": "ok"}
            at = self._get_at("src/dashboard/app.py")
            at.run()
            assert len(at.exception) == 0

    def test_overview_page_runs_without_exception(self, tmp_path, monkeypatch):
        from src.dashboard import data_loader

        csv_path = tmp_path / "observations.csv"
        _make_sample_df(50).to_csv(csv_path, index=False)
        monkeypatch.setattr(data_loader, "_RAW_CSV", csv_path)
        data_loader.load_observations.clear()

        eval_data_h = _make_eval_json("health")
        eval_data_r = _make_eval_json("restoration")
        (tmp_path / "evaluation_health.json").write_text(json.dumps(eval_data_h))
        (tmp_path / "evaluation_restoration.json").write_text(json.dumps(eval_data_r))
        monkeypatch.setattr(data_loader, "_EVAL_DIR", tmp_path)
        data_loader.load_evaluation.clear()

        with mock.patch("src.dashboard.api_client.APIClient") as MockClient:
            MockClient.return_value.health.return_value = {"status": "ok"}
            at = self._get_at("src/dashboard/pages/1_Overview.py")
            at.run()
            assert len(at.exception) == 0

    def test_mlops_status_page_runs_with_api_offline(self):
        from src.dashboard.api_client import APIError

        with mock.patch("src.dashboard.api_client.APIClient") as MockClient:
            MockClient.return_value.health.side_effect = APIError("offline")
            MockClient.return_value.model_info.side_effect = APIError("offline")
            at = self._get_at("src/dashboard/pages/7_MLOps_Status.py")
            at.run()
            assert len(at.exception) == 0

    def test_predict_page_runs_without_submission(self):
        """Predict page should render the form without submitting or calling API."""
        from src.dashboard.api_client import APIError

        with mock.patch("src.dashboard.api_client.APIClient") as MockClient:
            MockClient.return_value.health.side_effect = APIError("offline")
            at = self._get_at("src/dashboard/pages/5_Predict.py")
            at.run()
            assert len(at.exception) == 0

    def test_model_performance_page_with_missing_eval(self, tmp_path, monkeypatch):
        from src.dashboard import data_loader
        from src.dashboard.api_client import APIError

        # No eval files in tmp_path — page should degrade gracefully
        monkeypatch.setattr(data_loader, "_EVAL_DIR", tmp_path)
        data_loader.load_evaluation.clear()

        with mock.patch("src.dashboard.api_client.APIClient") as MockClient:
            MockClient.return_value.health.side_effect = APIError("offline")
            at = self._get_at("src/dashboard/pages/6_Model_Performance.py")
            at.run()
            assert len(at.exception) == 0

    def test_drift_page_shows_instructions_when_no_summary(self, isolated_reports_dir):
        """Page 8 shows instructions when drift_summary.json is absent."""
        from src.dashboard.api_client import APIError

        # isolated_reports_dir points the page at an EMPTY temporary reports
        # directory, so the "summary missing" branch is genuinely exercised
        # without deleting the real reports/drift_summary.json.
        assert not (isolated_reports_dir / "drift_summary.json").exists()
        page_path = "src/dashboard/pages/8_Drift_Monitoring.py"
        with mock.patch("src.dashboard.api_client.APIClient") as MockClient:
            MockClient.return_value.health.side_effect = APIError("offline")
            at = self._get_at(page_path)
            at.run()
            assert len(at.exception) == 0

    def test_drift_page_renders_with_summary(self, isolated_reports_dir):
        """Page 8 renders full drift results when drift_summary.json exists."""
        from src.dashboard.api_client import APIError

        drift_summary = {
            "generated_at": "2026-07-14T12:00:00+00:00",
            "shift_scale": 1.0,
            "reference_n": 100,
            "production_n": 100,
            "drift_threshold": 0.10,
            "feature_drift": {
                "drifted_count": 2,
                "total_columns": 15,
                "drifted_share": 0.13,
                "per_column": {
                    "water_temperature_c": {
                        "drifted": True,
                        "p_value": 0.001,
                        "method": "K-S p_value",
                    },
                    "turbidity_ntu": {"drifted": True, "p_value": 0.03, "method": "K-S p_value"},
                    "ph": {"drifted": False, "p_value": 0.45, "method": "K-S p_value"},
                },
            },
            "prediction_drift": {
                "health": {
                    "drifted": True,
                    "p_value": 0.002,
                    "statistic": 12.5,
                    "method": "chi2",
                    "reference_distribution": {"healthy": 0.5, "stressed": 0.3, "bleached": 0.2},
                    "current_distribution": {"healthy": 0.2, "stressed": 0.5, "bleached": 0.3},
                },
                "restoration": {
                    "drifted": False,
                    "p_value": 0.55,
                    "statistic": 1.2,
                    "method": "chi2",
                    "reference_distribution": {
                        "suitable": 0.6,
                        "moderately_suitable": 0.3,
                        "unsuitable": 0.1,
                    },
                    "current_distribution": {
                        "suitable": 0.55,
                        "moderately_suitable": 0.35,
                        "unsuitable": 0.1,
                    },
                },
            },
            "confidence_drift": {
                "health": {
                    "drifted": True,
                    "p_value": 0.001,
                    "statistic": 0.45,
                    "method": "ks",
                    "mean_reference": 0.92,
                    "mean_current": 0.71,
                    "delta": -0.21,
                },
                "restoration": {
                    "drifted": False,
                    "p_value": 0.32,
                    "statistic": 0.12,
                    "method": "ks",
                    "mean_reference": 0.88,
                    "mean_current": 0.85,
                    "delta": -0.03,
                },
            },
            "recommendation": "RETRAIN RECOMMENDED: Significant feature drift detected.",
            "synthetic_data_disclaimer": "Synthetic data only.",
        }
        # Write the fixture summary inside the isolated temporary reports
        # directory.  The real reports/drift_summary.json is never written to
        # and never deleted; pytest removes tmp_path automatically.
        summary_path = isolated_reports_dir / "drift_summary.json"
        summary_path.write_text(json.dumps(drift_summary))

        page_path = "src/dashboard/pages/8_Drift_Monitoring.py"
        with mock.patch("src.dashboard.api_client.APIClient") as MockClient:
            MockClient.return_value.health.side_effect = APIError("offline")
            at = self._get_at(page_path)
            at.run()
            assert len(at.exception) == 0


# ---------------------------------------------------------------------------
# Regression: AppTest script paths must be absolute and repo-root relative
# ---------------------------------------------------------------------------


class TestAppTestPathResolution:
    """
    Guards against the CI failure where ``AppTest.from_file("src/dashboard/…")``
    was resolved against ``tests/`` and looked for ``tests/src/dashboard/app.py``.
    """

    _SMOKE_SCRIPTS = (
        "src/dashboard/app.py",
        "src/dashboard/pages/1_Overview.py",
        "src/dashboard/pages/5_Predict.py",
        "src/dashboard/pages/6_Model_Performance.py",
        "src/dashboard/pages/7_MLOps_Status.py",
        "src/dashboard/pages/8_Drift_Monitoring.py",
    )

    def test_main_app_path_is_absolute(self):
        resolved = resolve_app_path("src/dashboard/app.py")
        assert resolved.is_absolute()
        assert resolved == _DASHBOARD_DIR / "app.py"

    @pytest.mark.parametrize("script", _SMOKE_SCRIPTS)
    def test_every_smoke_script_resolves_absolutely(self, script):
        resolved = resolve_app_path(script)
        assert resolved.is_absolute()
        assert resolved.is_file()
        assert resolved.is_relative_to(_DASHBOARD_DIR)

    @pytest.mark.parametrize("script", _SMOKE_SCRIPTS)
    def test_resolution_never_lands_under_tests_dir(self, script):
        """``tests/src/dashboard/...`` was the exact CI failure path."""
        resolved = resolve_app_path(script)
        assert not resolved.is_relative_to(_PROJECT_ROOT / "tests")

    def test_every_dashboard_page_resolves(self):
        pages = sorted(p.name for p in _PAGES_DIR.glob("*.py"))
        assert pages, "no dashboard pages found"
        for name in pages:
            assert resolve_app_path(f"src/dashboard/pages/{name}").is_file()

    def test_resolution_is_independent_of_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert resolve_app_path("src/dashboard/app.py") == _DASHBOARD_DIR / "app.py"
        assert resolve_app_path("src/dashboard/pages/8_Drift_Monitoring.py").is_file()

    def test_absolute_input_is_not_joined_twice(self):
        absolute = _PAGES_DIR / "8_Drift_Monitoring.py"
        assert resolve_app_path(absolute) == absolute
        assert resolve_app_path(str(absolute)) == absolute

    def test_path_and_str_inputs_agree(self):
        assert resolve_app_path("src/dashboard/app.py") == resolve_app_path(
            Path("src/dashboard/app.py")
        )

    def test_missing_script_raises_useful_error(self):
        with pytest.raises(AssertionError, match="dashboard script not found"):
            resolve_app_path("src/dashboard/pages/does_not_exist.py")

    def test_smoke_helper_hands_apptest_an_absolute_path(self, tmp_path, monkeypatch):
        """The helper used by every smoke test must never pass a relative path."""
        from streamlit.testing.v1 import AppTest

        monkeypatch.chdir(tmp_path)
        captured: list[str] = []

        def _fake_from_file(script_path, **kwargs):
            captured.append(str(script_path))
            return mock.MagicMock()

        monkeypatch.setattr(AppTest, "from_file", _fake_from_file)
        for script in self._SMOKE_SCRIPTS:
            TestAppSmoke()._get_at(script)

        assert len(captured) == len(self._SMOKE_SCRIPTS)
        for path in captured:
            assert Path(path).is_absolute()
            assert Path(path).is_file()
            assert not Path(path).is_relative_to(_PROJECT_ROOT / "tests")

    def test_main_app_runs_from_a_foreign_cwd(self, tmp_path, monkeypatch):
        """End-to-end proof: a real AppTest run does not depend on the cwd."""
        from src.dashboard import data_loader

        monkeypatch.setattr(data_loader, "_RAW_CSV", tmp_path / "none.csv")
        data_loader.load_observations.clear()
        foreign_cwd = tmp_path / "elsewhere"
        foreign_cwd.mkdir()
        monkeypatch.chdir(foreign_cwd)

        with mock.patch("src.dashboard.api_client.APIClient") as MockClient:
            MockClient.return_value.health.return_value = {"status": "ok"}
            at = TestAppSmoke()._get_at("src/dashboard/app.py")
            at.run()
            assert len(at.exception) == 0
