"""
Module 7: patient_pusher.py
----------------------------------
Creates (or attempts to create) a patient record in the destination platform
using the PATIENT_CREATE_URL endpoint provided in the project documentation.

This module purposefully keeps a very small public surface:

    PatientPusher.push_patient(patient: PatientData, **extra_fields)

It will return a tuple of (success: bool, response_json: dict, status_code: int).

Typical usage:

>>> from modules.patient_pusher import PatientPusher
>>> from modules.data_structures import PatientData
>>> patient = PatientData(patient_fname="John", patient_lname="Doe", dob="01/01/1980", medical_record_no="123ABC")
>>> pusher = PatientPusher()
>>> ok, resp, status = pusher.push_patient(patient, pgCompanyId="123", companyId="456")

If the call is successful (HTTP 200-299) ``ok`` will be ``True`` and ``resp`` will contain the
JSON response from the service (which is expected to include ``patientId``).

All requests automatically include the ``Authorization`` header using the
``AUTH_TOKEN`` environment variable.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Tuple, Dict, Any, Optional

import requests

from .data_structures import PatientData

# ---------------------------------------------------------------------------
# Configuration constants – source of truth is main.md but duplicated here so
# that the module is completely self-contained. If the project later introduces
# a central settings helper these can be replaced by imports.
# ---------------------------------------------------------------------------

TOKEN = os.getenv("AUTH_TOKEN")  # Must be set in environment (or .env)
PATIENT_CREATE_URL = (
    "https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net/api/Patient/create"
)

DEFAULT_TIMEOUT = 30  # seconds

logger = logging.getLogger(__name__)

if logger.level == logging.NOTSET:
    # Provide a minimal default configuration so that this module does not fail
    # silently if the application neglected to configure logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


class PatientPusher:
    """Thin wrapper around the Patient Create API endpoint."""

    def __init__(
        self,
        api_url: str = PATIENT_CREATE_URL,
        auth_token: Optional[str] = TOKEN,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not api_url:
            raise ValueError("api_url must be provided")

        self.api_url = api_url
        self.auth_token = auth_token or ""
        self.session = session or requests.Session()

        # Default headers – can be overridden per-request via kwargs in push_patient
        self.session.headers.update({"Content-Type": "application/json"})
        if self.auth_token:
            self.session.headers.update({"Authorization": f"Bearer {self.auth_token}"})

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    def push_patient(
        self,
        patient_data: PatientData | Dict[str, Any],
        **additional_fields: Any,
    ) -> Tuple[bool, Dict[str, Any], int]:
        """Send patient data to the Create Patient API.

        Parameters
        ----------
        patient_data : PatientData | Dict[str, Any]
            The core patient payload. If a *PatientData* object is provided it
            will be converted to *dict* via *to_dict()*.
        **additional_fields
            Arbitrary keyword arguments are merged into the payload. This allows callers
            to supply the *pgCompanyId*, *companyId*, *physicianGroup*, and
            *physicianGroupNPI* values that the endpoint expects.

        Returns
        -------
        Tuple[bool, dict, int]
            success flag, JSON response (or empty dict), HTTP status code.
        """

        if isinstance(patient_data, PatientData):
            payload: Dict[str, Any] = patient_data.to_dict()
        elif isinstance(patient_data, dict):
            payload = dict(patient_data)  # shallow copy
        else:
            raise TypeError("patient_data must be PatientData or dict")

        # Merge extra fields (they can overwrite existing keys by design)
        if additional_fields:
            payload.update(additional_fields)

        logger.debug("Patient creation payload: %s", json.dumps(payload, default=str))

        try:
            resp = self.session.post(
                self.api_url, json=payload, timeout=DEFAULT_TIMEOUT
            )
        except requests.RequestException as exc:
            logger.error("Patient push failed due to network error: %s", exc)
            return False, {"error": str(exc)}, -1

        status_ok = 200 <= resp.status_code < 300

        try:
            resp_json: Dict[str, Any] = resp.json() if resp.text else {}
        except ValueError:
            # Response not JSON – capture raw text for troubleshooting
            resp_json = {"raw": resp.text}

        if status_ok:
            logger.info(
                "Patient created successfully (status=%s): %s", resp.status_code, resp_json
            )
        else:
            logger.warning(
                "Patient creation returned status %s: %s", resp.status_code, resp_json
            )

        return status_ok, resp_json, resp.status_code

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def push_patient_if_not_exists(
        self,
        patient_data: PatientData | Dict[str, Any],
        exists: bool,
        **additional_fields: Any,
    ) -> Tuple[bool, Dict[str, Any], int]:
        """Conditionally create a patient only if it does not already exist.

        The *exists* flag should be the result of the MRN-based deduplication
        check performed upstream by *deduplication.py*.
        """

        if exists:
            logger.info("Patient already exists – skipping creation call")
            return True, {"message": "Patient already exists, not created."}, 200

        return self.push_patient(patient_data, **additional_fields) 