"""
src/dashboard/api_client.py — HTTP client for the CoralSense FastAPI service.

Only three endpoints are used by the dashboard:
  GET  /health
  GET  /model-info
  POST /predict/both

Configuration
-------------
Set CORALSENSE_API_URL to override the default base URL.
Set CORALSENSE_API_TIMEOUT (seconds) to override the default timeout.

Error handling
--------------
All network and HTTP errors are wrapped in APIError so that callers never
see raw tracebacks. Callers should catch APIError and display a friendly
message to the user.
"""

from __future__ import annotations

import os
from typing import Any

import requests

_DEFAULT_BASE_URL = "http://127.0.0.1:8000"
_DEFAULT_TIMEOUT = 10


class APIError(Exception):
    """Raised when the CoralSense API is unreachable or returns an error."""


class APIClient:
    """Reusable HTTP client for the CoralSense FastAPI inference service."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int | None = None,
    ) -> None:
        env_url = os.getenv("CORALSENSE_API_URL", _DEFAULT_BASE_URL)
        env_timeout = int(os.getenv("CORALSENSE_API_TIMEOUT", str(_DEFAULT_TIMEOUT)))
        self.base_url = (base_url or env_url).rstrip("/")
        self.timeout = timeout if timeout is not None else env_timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            r = requests.get(url, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                raise APIError(f"Unexpected response format from {path}.")
            return data
        except requests.Timeout as exc:
            raise APIError(
                f"Request to {path} timed out after {self.timeout}s. Is the API server running?"
            ) from exc
        except requests.ConnectionError as exc:
            raise APIError(
                f"Cannot connect to CoralSense API. Is uvicorn running at {self.base_url}?"
            ) from exc
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else "?"
            raise APIError(f"API returned HTTP {code} for {path}.") from exc
        except APIError:
            raise
        except Exception as exc:
            raise APIError(f"Unexpected error calling {path}: {type(exc).__name__}.") from exc

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            r = requests.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                raise APIError(f"Unexpected response format from {path}.")
            return data
        except requests.Timeout as exc:
            raise APIError(f"Request to {path} timed out after {self.timeout}s.") from exc
        except requests.ConnectionError as exc:
            raise APIError(
                f"Cannot connect to CoralSense API. Is uvicorn running at {self.base_url}?"
            ) from exc
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else "?"
            raise APIError(f"API returned HTTP {code} for {path}.") from exc
        except APIError:
            raise
        except Exception as exc:
            raise APIError(f"Unexpected error calling {path}: {type(exc).__name__}.") from exc

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """
        Call GET /health.

        Returns a dict with at least:
            status: "ok" | "degraded"
            health_model_ready: bool
            restoration_model_ready: bool
            timestamp: str
        """
        return self._get("/health")

    def model_info(self) -> dict[str, Any]:
        """
        Call GET /model-info.

        Returns safe champion metadata for both tasks.
        No internal paths or MLflow URIs are included.
        """
        return self._get("/model-info")

    def predict_both(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Call POST /predict/both.

        Parameters
        ----------
        payload:
            Dict matching ObservationInput schema.  Must contain the 16
            required inference features; must NOT contain reef_health or
            restoration_suitability (the API rejects extra fields).

        Returns
        -------
        dict with "health" and "restoration" PredictionResponse objects.
        """
        return self._post("/predict/both", payload)


def get_client() -> APIClient:
    """Return a default APIClient configured from environment variables."""
    return APIClient()
