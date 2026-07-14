"""
tests/test_api.py — Tests for the CoralSense FastAPI inference service (M9).

All tests use a fake ModelLoader so they never touch:
  - the canonical MLflow database (artifacts/mlruns.db)
  - real model files (models/best_model_*.joblib)
  - real preprocessors (data/processed/preprocessor_*.joblib)
  - the model registry (no registration or promotion)

Test isolation strategy
-----------------------
``FakeInferencePipeline`` produces deterministic outputs that mirror the
shape and field names of the real ``InferencePipeline``.
``FakeModelLoader`` implements the same interface as ``ModelLoader`` and is
injected via FastAPI's ``app.dependency_overrides``.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_loader
from src.api.schemas import (
    HEALTH_CLASSES,
    MAX_BATCH_SIZE,
    RESTORATION_CLASSES,
    VALID_REGIONS,
)

# ---------------------------------------------------------------------------
# Deterministic fake inference layer
# ---------------------------------------------------------------------------

_FAKE_DISCLAIMER = "Fake predictions for isolated testing. Synthetic data only."


class FakeInferencePipeline:
    """
    Deterministic stand-in for ``InferencePipeline``.

    Always predicts the first class with equal probabilities so tests
    can make exact assertions without training a real model.
    """

    def __init__(self, task: str) -> None:
        self.task = task
        self._classes: list[str] = sorted(
            HEALTH_CLASSES if task == "health" else RESTORATION_CLASSES
        )

    def predict_single(self, record: dict[str, Any]) -> dict[str, Any]:
        return self.predict_batch(pd.DataFrame([record]))[0]

    def predict_batch(self, raw_input: pd.DataFrame) -> list[dict[str, Any]]:
        n_rows = len(raw_input)
        n_classes = len(self._classes)
        even_p = round(1.0 / n_classes, 6)
        proba = {c: even_p for c in self._classes}
        return [
            {
                "predicted_class": self._classes[0],
                "probabilities": proba,
                "confidence": even_p,
                "task": self.task,
                "registered_model_name": f"coralsense_fake_{self.task}",
                "model_version": "1",
                "model_alias": "champion",
                "run_id": f"fake-run-{self.task}-abc123",
                "prediction_timestamp": "2024-01-15T10:00:00+00:00",
                "synthetic_data_disclaimer": _FAKE_DISCLAIMER,
            }
            for _ in range(n_rows)
        ]

    @property
    def label_names(self) -> list[str]:
        return self._classes

    @property
    def algo_name(self) -> str:
        return "fake_classifier"

    @property
    def model_version(self) -> str:
        return "1"

    @property
    def registered_model_name(self) -> str:
        return f"coralsense_fake_{self.task}"


class FakeModelLoader:
    """
    Test stand-in for ``ModelLoader``.

    ``health_ready`` and ``restoration_ready`` can be set to False to
    exercise 503 responses.
    """

    def __init__(self, health_ready: bool = True, restoration_ready: bool = True) -> None:
        self._health_ready = health_ready
        self._restoration_ready = restoration_ready
        self._health_pipeline = FakeInferencePipeline("health") if health_ready else None
        self._restoration_pipeline = (
            FakeInferencePipeline("restoration") if restoration_ready else None
        )

    @property
    def health_ready(self) -> bool:
        return self._health_ready

    @property
    def restoration_ready(self) -> bool:
        return self._restoration_ready

    @property
    def is_ready(self) -> bool:
        return self._health_ready and self._restoration_ready

    def get_health(self) -> FakeInferencePipeline:
        if self._health_pipeline is None:
            raise RuntimeError("Health model not available.")
        return self._health_pipeline

    def get_restoration(self) -> FakeInferencePipeline:
        if self._restoration_pipeline is None:
            raise RuntimeError("Restoration model not available.")
        return self._restoration_pipeline

    def model_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {}
        for task, ready, classes in [
            ("health", self._health_ready, sorted(HEALTH_CLASSES)),
            ("restoration", self._restoration_ready, sorted(RESTORATION_CLASSES)),
        ]:
            if not ready:
                info[task] = {"available": False}
            else:
                info[task] = {
                    "registered_model_name": f"coralsense_fake_{task}",
                    "version": "1",
                    "alias": "champion",
                    "algo_name": "fake_classifier",
                    "label_names": classes,
                    "run_id": f"fake-run-{task}",
                    "cv_macro_f1": 0.76,
                    "task": task,
                    "synthetic_data_disclaimer": _FAKE_DISCLAIMER,
                    "available": True,
                }
        return info


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Valid Gulf of Mannar observation used as a baseline in multiple tests.
_VALID_OBS: dict[str, Any] = {
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


@pytest.fixture()
def fake_loader() -> FakeModelLoader:
    return FakeModelLoader()


@pytest.fixture()
def unavailable_loader() -> FakeModelLoader:
    return FakeModelLoader(health_ready=False, restoration_ready=False)


@pytest.fixture()
def health_unavailable_loader() -> FakeModelLoader:
    return FakeModelLoader(health_ready=False, restoration_ready=True)


def _make_client(loader: FakeModelLoader) -> TestClient:
    """Return a TestClient with the given loader injected (no lifespan)."""
    app.dependency_overrides[get_loader] = lambda: loader
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def client(fake_loader: FakeModelLoader) -> TestClient:
    app.dependency_overrides[get_loader] = lambda: fake_loader
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


@pytest.fixture()
def client_no_models(unavailable_loader: FakeModelLoader) -> TestClient:
    app.dependency_overrides[get_loader] = lambda: unavailable_loader
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------


class TestRootEndpoint:
    def test_root_returns_200(self, client: TestClient) -> None:
        r = client.get("/")
        assert r.status_code == 200

    def test_root_contains_project_title(self, client: TestClient) -> None:
        r = client.get("/")
        data = r.json()
        assert "project" in data
        assert "Coral" in data["project"] or "coral" in data["project"].lower()

    def test_root_has_disclaimer(self, client: TestClient) -> None:
        data = client.get("/").json()
        assert "synthetic_data_disclaimer" in data
        assert len(data["synthetic_data_disclaimer"]) > 10

    def test_root_has_endpoints_list(self, client: TestClient) -> None:
        data = client.get("/").json()
        assert "endpoints" in data
        assert isinstance(data["endpoints"], list)
        assert len(data["endpoints"]) >= 7

    def test_root_has_docs_url(self, client: TestClient) -> None:
        data = client.get("/").json()
        assert "docs_url" in data


# ---------------------------------------------------------------------------
# Service health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200_when_ready(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_reports_both_models_ready(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["health_model_ready"] is True
        assert data["restoration_model_ready"] is True
        assert data["status"] == "ok"

    def test_health_returns_degraded_when_models_unavailable(
        self, client_no_models: TestClient
    ) -> None:
        data = client_no_models.get("/health").json()
        assert data["health_model_ready"] is False
        assert data["restoration_model_ready"] is False
        assert data["status"] == "degraded"

    def test_health_still_200_when_degraded(self, client_no_models: TestClient) -> None:
        # Health check must return 200 so orchestrators don't restart a degraded instance.
        r = client_no_models.get("/health")
        assert r.status_code == 200

    def test_health_has_timestamp(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert "timestamp" in data
        assert "T" in data["timestamp"]  # ISO 8601


# ---------------------------------------------------------------------------
# Model info endpoint
# ---------------------------------------------------------------------------


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client: TestClient) -> None:
        assert client.get("/model-info").status_code == 200

    def test_model_info_has_health_and_restoration(self, client: TestClient) -> None:
        data = client.get("/model-info").json()
        assert "health" in data
        assert "restoration" in data

    def test_model_info_health_has_safe_fields(self, client: TestClient) -> None:
        info = client.get("/model-info").json()["health"]
        for key in ("registered_model_name", "version", "alias", "algo_name", "label_names"):
            assert key in info, f"Missing key: {key}"

    def test_model_info_no_absolute_paths(self, client: TestClient) -> None:
        raw = client.get("/model-info").text
        assert "/home/" not in raw
        assert ".joblib" not in raw
        assert "mlruns.db" not in raw

    def test_model_info_has_disclaimer(self, client: TestClient) -> None:
        data = client.get("/model-info").json()
        assert "synthetic_data_disclaimer" in data


# ---------------------------------------------------------------------------
# Reef-health prediction
# ---------------------------------------------------------------------------


class TestReefHealthPrediction:
    def test_valid_observation_returns_200(self, client: TestClient) -> None:
        r = client.post("/predict/reef-health", json=_VALID_OBS)
        assert r.status_code == 200

    def test_response_has_required_fields(self, client: TestClient) -> None:
        data = client.post("/predict/reef-health", json=_VALID_OBS).json()
        for key in (
            "predicted_class",
            "probabilities",
            "confidence",
            "task",
            "registered_model_name",
            "model_version",
            "model_alias",
            "run_id",
            "prediction_timestamp",
            "synthetic_data_disclaimer",
        ):
            assert key in data, f"Missing response key: {key}"

    def test_predicted_class_is_valid_health_label(self, client: TestClient) -> None:
        data = client.post("/predict/reef-health", json=_VALID_OBS).json()
        assert data["predicted_class"] in HEALTH_CLASSES

    def test_probabilities_sum_to_one(self, client: TestClient) -> None:
        data = client.post("/predict/reef-health", json=_VALID_OBS).json()
        total = sum(data["probabilities"].values())
        assert math.isclose(total, 1.0, abs_tol=1e-4), f"Probability sum={total}"

    def test_probability_keys_are_valid_health_labels(self, client: TestClient) -> None:
        data = client.post("/predict/reef-health", json=_VALID_OBS).json()
        assert set(data["probabilities"].keys()).issubset(set(HEALTH_CLASSES))

    def test_confidence_matches_max_probability(self, client: TestClient) -> None:
        data = client.post("/predict/reef-health", json=_VALID_OBS).json()
        assert math.isclose(data["confidence"], max(data["probabilities"].values()), abs_tol=1e-6)

    def test_task_field_is_health(self, client: TestClient) -> None:
        data = client.post("/predict/reef-health", json=_VALID_OBS).json()
        assert data["task"] == "health"

    def test_has_synthetic_disclaimer(self, client: TestClient) -> None:
        data = client.post("/predict/reef-health", json=_VALID_OBS).json()
        assert "synthetic" in data["synthetic_data_disclaimer"].lower()

    def test_503_when_health_model_unavailable(
        self, health_unavailable_loader: FakeModelLoader
    ) -> None:
        app.dependency_overrides[get_loader] = lambda: health_unavailable_loader
        try:
            client = TestClient(app, raise_server_exceptions=False)
            r = client.post("/predict/reef-health", json=_VALID_OBS)
            assert r.status_code == 503
        finally:
            app.dependency_overrides.clear()

    def test_503_response_does_not_expose_internal_details(
        self, health_unavailable_loader: FakeModelLoader
    ) -> None:
        app.dependency_overrides[get_loader] = lambda: health_unavailable_loader
        try:
            client = TestClient(app, raise_server_exceptions=False)
            raw = client.post("/predict/reef-health", json=_VALID_OBS).text
            assert "/home/" not in raw
            assert "Traceback" not in raw
            assert "sqlite" not in raw.lower()
        finally:
            app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Restoration-suitability prediction
# ---------------------------------------------------------------------------


class TestRestorationPrediction:
    def test_valid_observation_returns_200(self, client: TestClient) -> None:
        assert client.post("/predict/restoration", json=_VALID_OBS).status_code == 200

    def test_predicted_class_is_valid_restoration_label(self, client: TestClient) -> None:
        data = client.post("/predict/restoration", json=_VALID_OBS).json()
        assert data["predicted_class"] in RESTORATION_CLASSES

    def test_probabilities_sum_to_one(self, client: TestClient) -> None:
        data = client.post("/predict/restoration", json=_VALID_OBS).json()
        total = sum(data["probabilities"].values())
        assert math.isclose(total, 1.0, abs_tol=1e-4)

    def test_probability_keys_are_valid_restoration_labels(self, client: TestClient) -> None:
        data = client.post("/predict/restoration", json=_VALID_OBS).json()
        assert set(data["probabilities"].keys()).issubset(set(RESTORATION_CLASSES))

    def test_task_field_is_restoration(self, client: TestClient) -> None:
        data = client.post("/predict/restoration", json=_VALID_OBS).json()
        assert data["task"] == "restoration"


# ---------------------------------------------------------------------------
# Combined prediction
# ---------------------------------------------------------------------------


class TestBothPrediction:
    def test_valid_observation_returns_200(self, client: TestClient) -> None:
        assert client.post("/predict/both", json=_VALID_OBS).status_code == 200

    def test_response_has_health_and_restoration(self, client: TestClient) -> None:
        data = client.post("/predict/both", json=_VALID_OBS).json()
        assert "health" in data
        assert "restoration" in data

    def test_health_result_has_valid_class(self, client: TestClient) -> None:
        data = client.post("/predict/both", json=_VALID_OBS).json()
        assert data["health"]["predicted_class"] in HEALTH_CLASSES

    def test_restoration_result_has_valid_class(self, client: TestClient) -> None:
        data = client.post("/predict/both", json=_VALID_OBS).json()
        assert data["restoration"]["predicted_class"] in RESTORATION_CLASSES

    def test_both_probabilities_sum_to_one(self, client: TestClient) -> None:
        data = client.post("/predict/both", json=_VALID_OBS).json()
        for task in ("health", "restoration"):
            total = sum(data[task]["probabilities"].values())
            assert math.isclose(total, 1.0, abs_tol=1e-4), f"{task} proba sum={total}"


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------


class TestBatchPrediction:
    def test_single_observation_batch_returns_200(self, client: TestClient) -> None:
        r = client.post("/predict/batch", json={"observations": [_VALID_OBS]})
        assert r.status_code == 200

    def test_batch_result_count_matches_input(self, client: TestClient) -> None:
        two_obs = [_VALID_OBS, _VALID_OBS]
        data = client.post("/predict/batch", json={"observations": two_obs}).json()
        assert data["count"] == 2
        assert len(data["results"]) == 2

    def test_batch_results_have_both_tasks(self, client: TestClient) -> None:
        data = client.post("/predict/batch", json={"observations": [_VALID_OBS]}).json()
        item = data["results"][0]
        assert "health" in item
        assert "restoration" in item

    def test_batch_health_labels_valid(self, client: TestClient) -> None:
        data = client.post("/predict/batch", json={"observations": [_VALID_OBS]}).json()
        for item in data["results"]:
            assert item["health"]["predicted_class"] in HEALTH_CLASSES

    def test_batch_restoration_labels_valid(self, client: TestClient) -> None:
        data = client.post("/predict/batch", json={"observations": [_VALID_OBS]}).json()
        for item in data["results"]:
            assert item["restoration"]["predicted_class"] in RESTORATION_CLASSES

    def test_batch_has_disclaimer(self, client: TestClient) -> None:
        data = client.post("/predict/batch", json={"observations": [_VALID_OBS]}).json()
        assert "synthetic_data_disclaimer" in data

    def test_batch_too_large_returns_422(self, client: TestClient) -> None:
        oversized = {"observations": [_VALID_OBS] * (MAX_BATCH_SIZE + 1)}
        r = client.post("/predict/batch", json=oversized)
        assert r.status_code == 422

    def test_empty_batch_returns_422(self, client: TestClient) -> None:
        r = client.post("/predict/batch", json={"observations": []})
        assert r.status_code == 422

    def test_batch_all_probability_rows_sum_to_one(self, client: TestClient) -> None:
        three_obs = [_VALID_OBS, _VALID_OBS, _VALID_OBS]
        data = client.post("/predict/batch", json={"observations": three_obs}).json()
        for item in data["results"]:
            for task in ("health", "restoration"):
                total = sum(item[task]["probabilities"].values())
                assert math.isclose(total, 1.0, abs_tol=1e-4)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_missing_required_field_returns_422(self, client: TestClient) -> None:
        obs = {k: v for k, v in _VALID_OBS.items() if k != "depth_m"}
        assert client.post("/predict/reef-health", json=obs).status_code == 422

    def test_extra_field_returns_422(self, client: TestClient) -> None:
        obs = {**_VALID_OBS, "reef_health": "healthy"}
        assert client.post("/predict/reef-health", json=obs).status_code == 422

    def test_target_label_as_input_returns_422(self, client: TestClient) -> None:
        obs = {**_VALID_OBS, "restoration_suitability": "suitable"}
        assert client.post("/predict/reef-health", json=obs).status_code == 422

    def test_wrong_type_for_numeric_field_returns_422(self, client: TestClient) -> None:
        obs = {**_VALID_OBS, "depth_m": "deep"}
        assert client.post("/predict/reef-health", json=obs).status_code == 422

    def test_nan_float_returns_422(self, client: TestClient) -> None:
        # JSON has no NaN literal; use a validator bypass attempt via string
        # Pydantic should reject non-finite string "NaN" as wrong type.
        obs = {**_VALID_OBS, "depth_m": "NaN"}
        assert client.post("/predict/reef-health", json=obs).status_code == 422

    def test_depth_out_of_range_low_returns_422(self, client: TestClient) -> None:
        obs = {**_VALID_OBS, "depth_m": -1.0}
        assert client.post("/predict/reef-health", json=obs).status_code == 422

    def test_depth_out_of_range_high_returns_422(self, client: TestClient) -> None:
        obs = {**_VALID_OBS, "depth_m": 51.0}
        assert client.post("/predict/reef-health", json=obs).status_code == 422

    def test_temperature_out_of_range_returns_422(self, client: TestClient) -> None:
        obs = {**_VALID_OBS, "water_temperature_c": 50.0}
        assert client.post("/predict/reef-health", json=obs).status_code == 422

    def test_ph_below_range_returns_422(self, client: TestClient) -> None:
        obs = {**_VALID_OBS, "ph": 5.0}
        assert client.post("/predict/reef-health", json=obs).status_code == 422

    def test_bleaching_percentage_over_100_returns_422(self, client: TestClient) -> None:
        obs = {**_VALID_OBS, "bleaching_percentage": 101.0}
        assert client.post("/predict/reef-health", json=obs).status_code == 422

    def test_coral_cover_negative_returns_422(self, client: TestClient) -> None:
        obs = {**_VALID_OBS, "coral_cover_percentage": -5.0}
        assert client.post("/predict/reef-health", json=obs).status_code == 422

    def test_invalid_region_returns_422(self, client: TestClient) -> None:
        obs = {**_VALID_OBS, "region": "Great Barrier Reef"}
        assert client.post("/predict/reef-health", json=obs).status_code == 422

    def test_all_four_valid_regions_accepted(self, client: TestClient) -> None:
        for region in VALID_REGIONS:
            obs = {**_VALID_OBS, "region": region}
            r = client.post("/predict/reef-health", json=obs)
            assert r.status_code == 200, f"Region {region!r} was incorrectly rejected"

    def test_latitude_out_of_range_returns_422(self, client: TestClient) -> None:
        obs = {**_VALID_OBS, "latitude": -100.0}
        assert client.post("/predict/reef-health", json=obs).status_code == 422

    def test_longitude_out_of_range_returns_422(self, client: TestClient) -> None:
        obs = {**_VALID_OBS, "longitude": 200.0}
        assert client.post("/predict/reef-health", json=obs).status_code == 422

    def test_sonar_backscatter_out_of_range_positive_returns_422(self, client: TestClient) -> None:
        obs = {**_VALID_OBS, "sonar_backscatter": 5.0}
        assert client.post("/predict/reef-health", json=obs).status_code == 422

    def test_optional_metadata_fields_accepted(self, client: TestClient) -> None:
        obs = {**_VALID_OBS, "latitude": 9.5, "longitude": 79.2, "timestamp": "2024-03-01"}
        assert client.post("/predict/reef-health", json=obs).status_code == 200

    def test_acoustic_complexity_above_1_returns_422(self, client: TestClient) -> None:
        obs = {**_VALID_OBS, "acoustic_complexity_index": 1.5}
        assert client.post("/predict/reef-health", json=obs).status_code == 422


# ---------------------------------------------------------------------------
# 503 model-unavailable responses
# ---------------------------------------------------------------------------


class TestModelUnavailable:
    def test_reef_health_returns_503_when_unavailable(self, client_no_models: TestClient) -> None:
        r = client_no_models.post("/predict/reef-health", json=_VALID_OBS)
        assert r.status_code == 503

    def test_restoration_returns_503_when_unavailable(self, client_no_models: TestClient) -> None:
        r = client_no_models.post("/predict/restoration", json=_VALID_OBS)
        assert r.status_code == 503

    def test_both_returns_503_when_unavailable(self, client_no_models: TestClient) -> None:
        r = client_no_models.post("/predict/both", json=_VALID_OBS)
        assert r.status_code == 503

    def test_batch_returns_503_when_unavailable(self, client_no_models: TestClient) -> None:
        r = client_no_models.post("/predict/batch", json={"observations": [_VALID_OBS]})
        assert r.status_code == 503

    def test_503_detail_is_user_friendly(self, client_no_models: TestClient) -> None:
        body = client_no_models.post("/predict/reef-health", json=_VALID_OBS).json()
        assert "detail" in body
        assert "Traceback" not in body.get("detail", "")
        assert "/home/" not in str(body)


# ---------------------------------------------------------------------------
# OpenAPI schema
# ---------------------------------------------------------------------------


class TestOpenAPI:
    def test_openapi_json_available(self, client: TestClient) -> None:
        r = client.get("/openapi.json")
        assert r.status_code == 200

    def test_openapi_json_is_valid(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        assert "openapi" in schema
        assert "paths" in schema
        assert "info" in schema

    def test_docs_url_returns_html(self, client: TestClient) -> None:
        r = client.get("/docs")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# Model loader singleton behaviour
# ---------------------------------------------------------------------------


class TestModelLoaderBehaviour:
    def test_fake_loader_is_ready(self, fake_loader: FakeModelLoader) -> None:
        assert fake_loader.is_ready is True

    def test_unavailable_loader_is_not_ready(self, unavailable_loader: FakeModelLoader) -> None:
        assert unavailable_loader.is_ready is False

    def test_model_info_excludes_paths(self, fake_loader: FakeModelLoader) -> None:
        info_str = str(fake_loader.model_info())
        assert ".joblib" not in info_str
        assert "/home/" not in info_str

    def test_model_info_health_has_label_names(self, fake_loader: FakeModelLoader) -> None:
        info = fake_loader.model_info()
        assert "label_names" in info["health"]
        assert len(info["health"]["label_names"]) == len(HEALTH_CLASSES)

    def test_model_info_restoration_has_label_names(self, fake_loader: FakeModelLoader) -> None:
        info = fake_loader.model_info()
        assert "label_names" in info["restoration"]
        assert len(info["restoration"]["label_names"]) == len(RESTORATION_CLASSES)

    def test_fake_pipeline_predict_single_returns_valid_class(
        self, fake_loader: FakeModelLoader
    ) -> None:
        pipeline = fake_loader.get_health()
        result = pipeline.predict_single(_VALID_OBS)
        assert result["predicted_class"] in HEALTH_CLASSES

    def test_fake_pipeline_predict_batch_length_matches_input(
        self, fake_loader: FakeModelLoader
    ) -> None:
        pipeline = fake_loader.get_restoration()
        df = pd.DataFrame([_VALID_OBS, _VALID_OBS, _VALID_OBS])
        results = pipeline.predict_batch(df)
        assert len(results) == 3

    def test_no_fit_or_fit_transform_called_during_prediction(self, client: TestClient) -> None:
        """
        Verify inference never calls fit/fit_transform.

        The FakeInferencePipeline has no fit method; if the real pipeline or
        any wrapper called fit, it would AttributeError.  This test proves
        the API layer does not call it.
        """
        # Patch a sentinel onto the fake pipeline
        called: dict[str, bool] = {}

        original_get_health = FakeModelLoader.get_health

        def patched_get_health(self):  # type: ignore[override]
            pipe = FakeInferencePipeline("health")
            pipe.fit = lambda *a, **kw: called.update({"fit": True})  # type: ignore[attr-defined]
            pipe.fit_transform = lambda *a, **kw: called.update({"fit_transform": True})  # type: ignore[attr-defined]
            return pipe

        FakeModelLoader.get_health = patched_get_health  # type: ignore[method-assign]
        try:
            r = client.post("/predict/reef-health", json=_VALID_OBS)
            assert r.status_code == 200
            assert "fit" not in called, "fit() was called during inference"
            assert "fit_transform" not in called, "fit_transform() was called during inference"
        finally:
            FakeModelLoader.get_health = original_get_health  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Probability and label integrity
# ---------------------------------------------------------------------------


class TestProbabilityIntegrity:
    def test_health_proba_keys_cover_all_classes(self, client: TestClient) -> None:
        data = client.post("/predict/reef-health", json=_VALID_OBS).json()
        proba_keys = set(data["probabilities"].keys())
        assert proba_keys == set(HEALTH_CLASSES)

    def test_restoration_proba_keys_cover_all_classes(self, client: TestClient) -> None:
        data = client.post("/predict/restoration", json=_VALID_OBS).json()
        proba_keys = set(data["probabilities"].keys())
        assert proba_keys == set(RESTORATION_CLASSES)

    def test_all_probabilities_non_negative(self, client: TestClient) -> None:
        data = client.post("/predict/reef-health", json=_VALID_OBS).json()
        for p in data["probabilities"].values():
            assert p >= 0.0, f"Negative probability: {p}"

    def test_confidence_equals_predicted_class_probability(self, client: TestClient) -> None:
        data = client.post("/predict/reef-health", json=_VALID_OBS).json()
        predicted = data["predicted_class"]
        proba = data["probabilities"]
        confidence = data["confidence"]
        assert math.isclose(confidence, proba[predicted], abs_tol=1e-6)

    def test_restoration_proba_keys_cover_all_classes_in_batch(self, client: TestClient) -> None:
        data = client.post("/predict/batch", json={"observations": [_VALID_OBS]}).json()
        proba_keys = set(data["results"][0]["restoration"]["probabilities"].keys())
        assert proba_keys == set(RESTORATION_CLASSES)
