"""
Utilities for extracting information from customer messages.

This module defines:

    extract_customer_information: Interprets customer messages using an
        LLM and validates policyholder identity information, caller role,
        representative details, intent, and case hints.
"""

import json
import re
from collections.abc import Callable
from typing import Any

from agent.prompts import EXTRACTION_PROMPT
from agent.state import ConversationState


IDENTITY_FIELDS = {
    "name",
    "dob",
    "phone",
    "email",
    "id_last4",
    "policy_number",
}

CALLER_FIELDS = {
    "caller_role",
    "representative_name",
    "representative_relationship",
}

ALLOWED_CALLER_ROLES = {
    "policyholder",
    "representative",
}

ALLOWED_INTENTS = {
    "denial_question",
    "status_inquiry",
    "document_submission",
    "payment_question",
    "next_steps",
    "general_claim_question",
    "portal_support",
}

ALLOWED_CASE_TYPES = {
    "healthcare",
    "dental",
    "auto",
}

ALLOWED_EMOTIONS = {
    "frustration",
    "anxiety",
    "anger",
    "confusion",
    "refusal",
}

ALLOWED_SCOPES = {
    "in_scope",
    "out_of_scope",
    "mixed",
    "unclear",
}

MONTHS = {
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
}

CASE_HINT_FIELDS = {
    "intent",
    "case_type",
    "month",
    "year",
    "reported_status",
    "case_id",
}

ALLOWED_CLEAR_FIELDS = (
    CASE_HINT_FIELDS
    | CALLER_FIELDS
    | {
        f"identity_fields.{name}"
        for name in IDENTITY_FIELDS
    }
)

EXTRACTION_FIELDS = {
    "identity_fields",
    *CALLER_FIELDS,
    *CASE_HINT_FIELDS,
    "emotion",
    "scope",
    "human_requested",
    "clarification_question",
    "clear_fields",
}


def _recover_explicit_policyholder_pii(
    caller_role: str | None,
    message: str,
    identity_fields: dict[str, str],
) -> None:
    """
    Recover policyholder PII explicitly present in the current message.

    This does not guess, verify, or retrieve values from fixtures.
    It only fills values clearly written by the customer.
    """

    normalized = message.lower()

    if caller_role == "representative":
        ssn_pattern = (
            r"\b(?:her|his|their|the policyholder(?:'s)?)\s+"
            r"(?:ssn|social security)"
            r"\s+(?:last\s*(?:four|4)|last4)"
            r"\s*(?:is|:|-)?\s*(\d{4})\b"
        )
    else:
        ssn_pattern = (
            r"\b(?:my\s+)?(?:ssn|social security)"
            r"\s+(?:last\s*(?:four|4)|last4)"
            r"\s*(?:is|:|-)?\s*(\d{4})\b"
        )

    ssn_match = re.search(ssn_pattern, normalized)

    if ssn_match:
        identity_fields.setdefault(
            "id_last4",
            ssn_match.group(1),
        )

    policy_match = re.search(
        r"\bpolicy(?:\s+number)?"
        r"\s*(?:is|:|-)?\s*(pol-\d+)\b",
        normalized,
    )

    if policy_match:
        identity_fields.setdefault(
            "policy_number",
            policy_match.group(1).upper(),
        )

    if caller_role == "representative":
        dob_pattern = (
            r"\b(?:her|his|their|the policyholder(?:'s)?)\s+"
            r"(?:dob|date of birth)"
            r"\s*(?:is|:|-)?\s*"
            r"(\d{4}-\d{2}-\d{2})\b"
        )
    else:
        dob_pattern = (
            r"\b(?:my\s+)?(?:dob|date of birth)"
            r"\s*(?:is|:|-)?\s*"
            r"(\d{4}-\d{2}-\d{2})\b"
        )

    dob_match = re.search(dob_pattern, normalized)

    if dob_match:
        identity_fields.setdefault(
            "dob",
            dob_match.group(1),
        )


def extract_customer_information(
    conversation: ConversationState,
    message: str,
    ask_model: Callable[[str], str],
) -> dict[str, Any]:
    """
    Extract customer information using conversation context.

    Args:
        conversation: Current conversation state. Its messages contain
            earlier turns, excluding the current customer message.
        message: Current customer message.
        ask_model: Function accepting a prompt and returning JSON text.

    Returns:
        Validated information extracted from the current message.

    Raises:
        ValueError: If the model output has an invalid structure or value.
    """

    context = {
        "recent_messages": conversation.messages[-12:],
        "caller_context": {
            "caller_role": conversation.caller_role.value,
            "representative_name": conversation.representative_name,
            "representative_relationship": (
                conversation.representative_relationship
            ),
        },
        "remembered_intent": conversation.remembered_intent,
        "case_hints": {
            "case_type": conversation.remembered_case_type,
            "month": conversation.remembered_month,
            "year": conversation.remembered_year,
            "reported_status": conversation.remembered_status_hint,
            "case_id": conversation.remembered_case_id,
        },
        "current_message": message,
    }

    raw_response = ask_model(
        EXTRACTION_PROMPT
        + json.dumps(context, ensure_ascii=False)
    )

    try:
        extracted = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(
            "The extractor did not return valid JSON."
        ) from error

    if not isinstance(extracted, dict):
        raise ValueError(
            "The extraction result must be a JSON object."
        )

    if set(extracted) != EXTRACTION_FIELDS:
        raise ValueError(
            "The extraction result has unexpected or missing fields."
        )

    identity = extracted["identity_fields"]

    if not isinstance(identity, dict):
        raise ValueError("identity_fields must be an object.")

    # Use the role from this extraction, because conversation.caller_role
    # has not been updated yet when this function is first called.
    _recover_explicit_policyholder_pii(
        caller_role=extracted["caller_role"],
        message=message,
        identity_fields=identity,
    )

    # identity_fields contains policyholder PII, not representative PII.
    for name, value in identity.items():
        if name not in IDENTITY_FIELDS:
            raise ValueError(
                f"Unsupported identity field: {name}"
            )

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Invalid identity value: {name}"
            )

    choices = {
        "caller_role": ALLOWED_CALLER_ROLES,
        "intent": ALLOWED_INTENTS,
        "case_type": ALLOWED_CASE_TYPES,
        "month": MONTHS,
        "emotion": ALLOWED_EMOTIONS,
    }

    for name, allowed_values in choices.items():
        value = extracted[name]

        if value is not None:
            if (
                not isinstance(value, str)
                or value not in allowed_values
            ):
                raise ValueError(
                    f"Invalid extraction value: {name}"
                )

    scope = extracted["scope"]

    if (
        not isinstance(scope, str)
        or scope not in ALLOWED_SCOPES
    ):
        raise ValueError("Invalid scope value.")

    year = extracted["year"]

    if year is not None:
        if type(year) is not int or not 1 <= year <= 9999:
            raise ValueError(
                "year must be a valid integer year."
            )

    text_fields = (
        "representative_name",
        "representative_relationship",
        "reported_status",
        "case_id",
        "clarification_question",
    )

    for name in text_fields:
        value = extracted[name]

        if value is not None:
            if (
                not isinstance(value, str)
                or not value.strip()
            ):
                raise ValueError(
                    f"Invalid extraction value: {name}"
                )

    if type(extracted["human_requested"]) is not bool:
        raise ValueError(
            "human_requested must be a boolean."
        )

    clear_fields = extracted["clear_fields"]

    if not isinstance(clear_fields, list):
        raise ValueError("clear_fields must be a list.")

    normalized_clear_fields = []

    for name in clear_fields:
        if (
            not isinstance(name, str)
            or name not in ALLOWED_CLEAR_FIELDS
        ):
            raise ValueError(
                "Unsupported field retraction."
            )

        if name.startswith("identity_fields."):
            field_name = name.split(".", 1)[1]
            has_replacement = field_name in identity
        else:
            has_replacement = extracted[name] is not None

        # A supplied replacement already overwrites the old value.
        if has_replacement:
            continue

        if name not in normalized_clear_fields:
            normalized_clear_fields.append(name)

    extracted["clear_fields"] = normalized_clear_fields

    return extracted