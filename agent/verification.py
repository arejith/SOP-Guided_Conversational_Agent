"""
Identity verification for the SOP-guided insurance support agent.

This module defines:

    IdentityVerifier: Verifies customer identity against policyholder
        fixtures using at least three matching PII fields.
"""

import re
from datetime import date, datetime
from typing import Any

from services.data_service import InsuranceDataService


class IdentityVerifier:
    """
    Verify customer identity using policyholder fixture records.

    Attributes:
        data_service: Service providing policyholder fixture records.
    """

    PII_FIELDS = (
        "name",
        "dob",
        "phone",
        "email",
        "id_last4",
    )

    ALIAS_FIELDS = {
        "name": "name_aliases",
        "phone": "phone_aliases",
        "email": "email_aliases",
    }

    def __init__(
        self,
        data_service: InsuranceDataService,
    ) -> None:
        """
        Initialize the identity verifier.

        Args:
            data_service: Service used to access policyholder fixtures.
        """

        self.data_service = data_service

    def find_verified_policyholder(
        self,
        identity_fields: dict[str, str],
    ) -> dict[str, Any] | None:
        """
        Find exactly one policyholder matching at least three PII fields.

        A supplied policy number narrows the search but does not count
        toward the required three PII fields.

        Args:
            identity_fields: Identity information supplied by the customer.

        Returns:
            The matching policyholder record, or None if verification
            fails or multiple records qualify.
        """

        provided_policy = identity_fields.get("policy_number")
        normalized_policy = None

        if provided_policy is not None:
            normalized_policy = self._normalize_value(
                "policy_number",
                provided_policy,
            )

            if normalized_policy is None:
                return None

        candidates = []

        for policyholder in self.data_service.policyholders:
            if normalized_policy is not None:
                stored_policy = self._normalize_value(
                    "policy_number",
                    policyholder.get("policy_number"),
                )

                if normalized_policy != stored_policy:
                    continue

            if self._matches_three_pii_fields(
                identity_fields,
                policyholder,
            ):
                candidates.append(policyholder)

        if len(candidates) != 1:
            return None

        return candidates[0]

    def _matches_three_pii_fields(
        self,
        identity_fields: dict[str, str],
        policyholder: dict[str, Any],
    ) -> bool:
        """
        Check whether three distinct permitted PII fields match.

        Args:
            identity_fields: Identity information supplied by the customer.
            policyholder: Candidate policyholder fixture record.

        Returns:
            True when at least three distinct PII fields match.
        """

        matching_fields = 0

        for field_name in self.PII_FIELDS:
            # The SOP permits SSN last four, not national ID last four.
            if field_name == "id_last4":
                if policyholder.get("id_type") != "ssn_last4":
                    continue

            provided_value = self._normalize_value(
                field_name,
                identity_fields.get(field_name),
            )

            if provided_value is None:
                continue

            stored_values = [policyholder.get(field_name)]

            alias_field = self.ALIAS_FIELDS.get(field_name)

            if alias_field is not None:
                aliases = policyholder.get(alias_field, [])

                if isinstance(aliases, list):
                    stored_values.extend(aliases)

            field_matches = any(
                provided_value == self._normalize_value(field_name, stored_value)
                for stored_value in stored_values
            )

            # Matching an alias and the primary value still counts once.
            if field_matches:
                matching_fields += 1

        return matching_fields >= 3

    @staticmethod
    def _normalize_value(
        field_name: str,
        value: Any,
    ) -> str | None:
        """
        Normalize a value according to its identity field.

        Args:
            field_name: Identity field being compared.
            value: Raw customer or fixture value.

        Returns:
            Normalized value, or None for an unsupported or invalid value.
        """

        if not isinstance(value, str):
            return None

        value = value.strip()

        if not value:
            return None

        if field_name == "name":
            return " ".join(value.casefold().split())

        if field_name == "email":
            # Preserve punctuation so different addresses stay distinct.
            if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value):
                return None

            return value.casefold()

        if field_name == "phone":
            # Accept common formatting, but require the country code
            # when the fixture includes it. Do not guess a country.
            if not re.fullmatch(r"\+?[0-9\s().-]+", value):
                return None

            digits = re.sub(r"[^0-9]", "", value)

            if not 8 <= len(digits) <= 15:
                return None

            return digits

        if field_name == "dob":
            # Unambiguous English month names are accepted without guessing
            # the locale of numeric dates such as 03/04/1985.
            natural = re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", value, flags=re.I)
            natural = " ".join(natural.replace(",", " ").split())
            # Models sometimes return ISO dates with an ordinal suffix or
            # month-first punctuation; normalize those before strict parsing.
            natural = (
                natural.replace("-", " ")
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
                else natural
            )
            iso_prefix = re.fullmatch(
                r"(\d{4}-\d{2}-\d{2})(?:[\sTtZz0-9:.,/+\-]+)?", value
            )
            if iso_prefix:
                try:
                    return date.fromisoformat(iso_prefix.group(1)).isoformat()
                except ValueError:
                    return None
            for pattern in (
                "%B %d %Y",
                "%b %d %Y",
                "%d %B %Y",
                "%d %b %Y",
                "%Y %m %d",
            ):
                try:
                    return datetime.strptime(natural, pattern).date().isoformat()
                except ValueError:
                    pass
            if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
                return None

            try:
                return date.fromisoformat(value).isoformat()
            except ValueError:
                return None

        if field_name == "id_last4":
            if re.fullmatch(r"[0-9]{4}", value):
                return value

            return None

        if field_name == "policy_number":
            return value.upper()

        return None
