"""
Module 5: validator.py
Purpose: Centralized validation for patient and order data extracted by previous modules.
Ensures that all required critical fields are present and that values follow the expected
format (e.g. dates, sex).  Designed to fail fast and provide detailed error messages that
can be logged or written to output reports.
"""

from __future__ import annotations

import re
import logging
from typing import List, Tuple, Dict

from .data_structures import (
    PatientData,
    OrderData,
    ParsedResult,
    get_critical_patient_fields,
    get_critical_order_fields,
)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
# Accept strictly MM/DD/YYYY (zero-padded month and day)
_DATE_REGEX = re.compile(r"^(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/[0-9]{4}$")

# Allowed values for the patient_sex field
_ALLOWED_SEX_VALUES = {"MALE", "FEMALE"}


class DataValidator:
    """Validate :class:`PatientData` and :class:`OrderData` objects.

    Example
    -------
    >>> validator = DataValidator()
    >>> ok, errors = validator.validate_patient(patient_data)
    >>> ok, errors = validator.validate_order(order_data)
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    # ---------------------------------------------------------------------
    # Low-level helpers
    # ---------------------------------------------------------------------
    def _is_valid_date(self, date_str: str) -> bool:
        """Return True if *date_str* matches ``MM/DD/YYYY`` pattern."""
        if not date_str:
            return False
        return bool(_DATE_REGEX.match(date_str.strip()))

    # ---------------------------------------------------------------------
    # High-level validation routines
    # ---------------------------------------------------------------------
    def validate_patient(self, patient: PatientData) -> Tuple[bool, List[str]]:
        """Validate a :class:`PatientData` instance.

        Returns
        -------
        Tuple[bool, List[str]]
            *bool*  – Overall validity flag.
            *List[str]* – List of error messages (empty when valid).
        """
        errors: List[str] = []
        patient_dict: Dict = patient.to_dict()

        # Required critical fields
        for field in get_critical_patient_fields():
            if not str(patient_dict.get(field, "")).strip():
                errors.append(f"Missing patient field: '{field}'")

        # Validate sex value
        sex_val = str(patient.patient_sex).strip().upper() if patient.patient_sex else ""
        if sex_val and sex_val not in _ALLOWED_SEX_VALUES:
            errors.append(
                f"Invalid patient_sex '{patient.patient_sex}' (allowed: MALE/FEMALE)"
            )

        # Validate DOB format
        if patient.dob and not self._is_valid_date(patient.dob):
            errors.append(
                f"Invalid DOB format '{patient.dob}' (expected MM/DD/YYYY)"
            )

        is_valid = not errors
        return is_valid, errors

    def validate_order(self, order: OrderData) -> Tuple[bool, List[str]]:
        """Validate an :class:`OrderData` instance."""
        errors: List[str] = []
        order_dict: Dict = order.to_dict()

        # Required critical fields
        for field in get_critical_order_fields():
            if field == "episode_diagnoses":
                if len(order.episode_diagnoses) == 0:
                    errors.append("Missing order field: 'episode_diagnoses'")
            else:
                if not str(order_dict.get(field, "")).strip():
                    errors.append(f"Missing order field: '{field}'")

        # Validate important date fields (if present)
        date_fields = [
            "order_date",
            "episode_start_date",
            "episode_end_date",
            "start_of_care",
            "signed_by_physician_date",
        ]
        for field in date_fields:
            value = str(order_dict.get(field, "")).strip()
            if value and not self._is_valid_date(value):
                errors.append(
                    f"Invalid date format in '{field}': '{value}' (expected MM/DD/YYYY)"
                )

        is_valid = not errors
        return is_valid, errors

    def validate_parsed_result(self, result: ParsedResult) -> Tuple[bool, List[str]]:
        """Validate a :class:`ParsedResult` (combined patient + order).

        This is a convenience wrapper around :py:meth:`validate_patient` and
        :py:meth:`validate_order`.  It aggregates error messages from both
        routines and returns a single validity flag.
        """
        errors: List[str] = []

        patient_valid, patient_errors = self.validate_patient(result.patient_data)
        order_valid, order_errors = self.validate_order(result.order_data)

        errors.extend(patient_errors)
        errors.extend(order_errors)

        is_valid = patient_valid and order_valid
        if not is_valid:
            self.logger.warning(
                "Validation failed for doc_id %s – %s",
                result.doc_id,
                "; ".join(errors),
            )
        return is_valid, errors


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def validate_results(
    results: List[ParsedResult],
) -> Tuple[List[ParsedResult], List[Dict[str, object]]]:
    """Split *results* into valid and invalid based on :class:`DataValidator`.

    Returns a tuple ``(valid_results, invalid_details)`` where *invalid_details*
    is a list of dictionaries ``{"doc_id": ..., "errors": [...]}``.
    """
    validator = DataValidator()
    valid: List[ParsedResult] = []
    invalid: List[Dict[str, object]] = []

    for res in results:
        ok, errs = validator.validate_parsed_result(res)
        if ok:
            valid.append(res)
        else:
            invalid.append({"doc_id": res.doc_id, "errors": errs})
    return valid, invalid 