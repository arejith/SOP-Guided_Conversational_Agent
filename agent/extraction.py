"""
Utilities for extracting information from customer messages.

This module defines:

    extract_customer_information: Interprets customer messages using an
        LLM and validates the extracted information.
"""

import json
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

ALLOWED_INTENTS = {
    "denial_question",
    "status_inquiry",
    "document_submission",
    "payment_question",
    "next_steps",
    "general_claim_question",
    "portal_support",
}

ALLOWED_CASE_TYPES = {"healthcare", "dental", "auto"}

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

ALLOWED_CLEAR_FIELDS = CASE_HINT_FIELDS | {
    f"identity_fields.{name}" for name in IDENTITY_FIELDS
}

EXTRACTION_FIELDS = {
    "identity_fields",
    *CASE_HINT_FIELDS,
    "emotion",
    "scope",
    "human_requested",
    "clarification_question",
    "clear_fields",
}


def extract_customer_information(
    conversation: ConversationState,
    message: str,
    ask_model: Callable[[str], str],
) -> dict[str, Any]:
    """
    Extract information from a customer message using conversation context.

    Args:
        conversation: Current conversation state. Its messages contain
            earlier turns, excluding the current customer message.
        message: Current customer message.
        ask_model: Function accepting a prompt and returning JSON text.

    Returns:
        Validated information extracted from the customer message.

    Raises:
        ValueError: If the model output has an invalid structure or value.
    """

    context = {
        "recent_messages": conversation.messages[-12:],
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
        EXTRACTION_PROMPT + json.dumps(context, ensure_ascii=False)
    )

    try:
        extracted = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(
            "The extractor did not return valid JSON."
        ) from error

    if not isinstance(extracted, dict):
        raise ValueError("The extraction result must be a JSON object.")

    if set(extracted) != EXTRACTION_FIELDS:
        raise ValueError(
            "The extraction result has unexpected or missing fields."
        )

    identity = extracted["identity_fields"]

    if not isinstance(identity, dict):
        raise ValueError("identity_fields must be an object.")

    for name, value in identity.items():
        if name not in IDENTITY_FIELDS:
            raise ValueError(f"Unsupported identity field: {name}")

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid identity value: {name}")

    choices = {
        "intent": ALLOWED_INTENTS,
        "case_type": ALLOWED_CASE_TYPES,
        "month": MONTHS,
        "emotion": ALLOWED_EMOTIONS,
    }

    for name, allowed_values in choices.items():
        value = extracted[name]

        if value is not None:
            if not isinstance(value, str) or value not in allowed_values:
                raise ValueError(f"Invalid extraction value: {name}")

    scope = extracted["scope"]

    if not isinstance(scope, str) or scope not in ALLOWED_SCOPES:
        raise ValueError("Invalid scope value.")

    year = extracted["year"]

    if year is not None:
        if type(year) is not int or not 1 <= year <= 9999:
            raise ValueError("year must be a valid integer year.")

    for name in (
        "reported_status",
        "case_id",
        "clarification_question",
    ):
        value = extracted[name]

        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Invalid extraction value: {name}")

    if type(extracted["human_requested"]) is not bool:
        raise ValueError("human_requested must be a boolean.")

    clear_fields = extracted["clear_fields"]

    if not isinstance(clear_fields, list):
        raise ValueError("clear_fields must be a list.")

    normalized_clear_fields = []

    for name in clear_fields:
        if not isinstance(name, str) or name not in ALLOWED_CLEAR_FIELDS:
            raise ValueError("Unsupported field retraction.")

        if name.startswith("identity_fields."):
            field_name = name.split(".", 1)[1]
            has_replacement = field_name in identity
        else:
            has_replacement = extracted[name] is not None

        # A supplied replacement already overwrites the old value.
        # Clearing that same field is redundant.
        if has_replacement:
            continue

        if name not in normalized_clear_fields:
            normalized_clear_fields.append(name)

    extracted["clear_fields"] = normalized_clear_fields

    return extracted