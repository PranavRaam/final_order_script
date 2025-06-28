"""
Module 8: order_pusher.py
----------------------------------
Pushes order (episode) details to the ORDER_PUSH_URL endpoint.

API Contract (based on earlier prototype code):
• Requires JSON payload containing – at minimum – patientId, companyId, pgCompanyId, orderNo, episodeStartDate, episodeEndDate, startOfCare.
• Endpoint returns 201 on create, 409 on duplicate (already exists), other codes on error.

Public interface:
    OrderPusher.push_order(order: OrderData, patient_id: str, **extra_fields)

Returns (success, response_json, status_code)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Any, Tuple, Optional

import requests

from .data_structures import OrderData

TOKEN = os.getenv("AUTH_TOKEN")
ORDER_PUSH_URL = (
    "https://dawavorderpatient-hqe2apddbje9gte0.eastus-01.azurewebsites.net/api/Order"
)
DEFAULT_TIMEOUT = 30

logger = logging.getLogger(__name__)
if logger.level == logging.NOTSET:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class OrderPusher:
    """Wrapper around the Order push API."""

    def __init__(
        self,
        api_url: str = ORDER_PUSH_URL,
        auth_token: Optional[str] = TOKEN,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not api_url:
            raise ValueError("api_url must be provided")
        self.api_url = api_url
        self.auth_token = auth_token or ""
        self.session = session or requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if self.auth_token:
            self.session.headers.update({"Authorization": f"Bearer {self.auth_token}"})

    def push_order(
        self,
        order_data: OrderData | Dict[str, Any],
        patient_id: str,
        **additional_fields: Any,
    ) -> Tuple[bool, Dict[str, Any], int]:
        """Send order data to the API.

        The *patient_id* is mandatory; it will be injected into the payload.
        Extra kwargs let callers supply companyId, pgCompanyId, etc.
        """
        if not patient_id:
            raise ValueError("patient_id must be provided")

        if isinstance(order_data, OrderData):
            payload: Dict[str, Any] = order_data.to_dict()
        elif isinstance(order_data, dict):
            payload = dict(order_data)
        else:
            raise TypeError("order_data must be OrderData or dict")

        payload["patientId"] = patient_id
        if additional_fields:
            payload.update(additional_fields)

        logger.debug("Order push payload: %s", json.dumps(payload, default=str))

        try:
            resp = self.session.post(self.api_url, json=payload, timeout=DEFAULT_TIMEOUT)
        except requests.RequestException as exc:
            logger.error("Order push failed due to network error: %s", exc)
            return False, {"error": str(exc)}, -1

        status_ok = resp.status_code == 201 or resp.status_code == 409 or (200 <= resp.status_code < 300)
        try:
            resp_json = resp.json() if resp.text else {}
        except ValueError:
            resp_json = {"raw": resp.text}

        if resp.status_code == 201:
            logger.info("Order created successfully (status=201)")
        elif resp.status_code == 409:
            logger.warning("Duplicate order (status=409)")
        elif 200 <= resp.status_code < 300:
            logger.info("Order request returned success status %s", resp.status_code)
        else:
            logger.error("Order push returned status %s: %s", resp.status_code, resp_json)

        return status_ok, resp_json, resp.status_code 