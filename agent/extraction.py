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
from agent.verification import IdentityVerifier
from services.errors import ModelResponseError

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
    | {f"identity_fields.{name}" for name in IDENTITY_FIELDS}
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
    "identity_ambiguous",
    "identity_context_changed",
    "general_question",
    "conversation_action",
}


def extraction_schema() -> dict[str, Any]:
    """Describe structured extraction; semantic validation still follows."""

    properties = {name: {"type": ["string", "null"]} for name in EXTRACTION_FIELDS}
    properties.update(
        {
            "identity_fields": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    name: {"type": ["string", "null"]}
                    for name in sorted(IDENTITY_FIELDS)
                },
                "required": sorted(IDENTITY_FIELDS),
            },
            "clear_fields": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(ALLOWED_CLEAR_FIELDS)},
            },
            "year": {"type": ["integer", "null"]},
            "human_requested": {"type": "boolean"},
            "identity_ambiguous": {"type": "boolean"},
            "identity_context_changed": {"type": "boolean"},
        }
    )
    for name, choices in {
        "caller_role": ALLOWED_CALLER_ROLES,
        "intent": ALLOWED_INTENTS,
        "case_type": ALLOWED_CASE_TYPES,
        "month": MONTHS,
        "emotion": ALLOWED_EMOTIONS,
        "general_question": {
            "verification_reason",
            "identity_options",
            "document_preparation",
        },
    }.items():
        properties[name]["enum"] = sorted(choices) + [None]
    properties["scope"] = {"type": "string", "enum": sorted(ALLOWED_SCOPES)}
    properties["conversation_action"] = {
        "type": "string",
        "enum": ["continue", "wrap_up"],
    }
    return {
        "type": "object",
        "properties": properties,
        "required": sorted(EXTRACTION_FIELDS),
        "additionalProperties": False,
    }


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
            "representative_relationship": (conversation.representative_relationship),
        },
        "remembered_intent": conversation.remembered_intent,
        "case_hints": {
            "case_type": conversation.remembered_case_type,
            "month": conversation.remembered_month,
            "year": conversation.remembered_year,
            "reported_status": conversation.remembered_status_hint,
            "case_id": conversation.remembered_case_id,
        },
        "pending_question": conversation.pending_question,
        "pending_field": conversation.pending_field,
        "supplied_identity_fields": conversation.identity_fields,
        "phase": conversation.phase.value,
        "current_message": message,
    }

    raw_response = ask_model(
        EXTRACTION_PROMPT + json.dumps(context, ensure_ascii=False)
    )

    try:
        extracted = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as error:
        raise ModelResponseError("The extractor did not return valid JSON.") from error

    if not isinstance(extracted, dict):
        raise ModelResponseError("The extraction result must be a JSON object.")

    if set(extracted) != EXTRACTION_FIELDS:
        raise ModelResponseError(
            "The extraction result has unexpected or missing fields."
        )

    identity = extracted["identity_fields"]

    if not isinstance(identity, dict):
        raise ModelResponseError("identity_fields must be an object.")
    if not set(identity).issubset(IDENTITY_FIELDS):
        raise ModelResponseError("Unsupported identity field.")

    # Structured Outputs requires all schema fields, with null for omissions.
    identity = {name: value for name, value in identity.items() if value is not None}
    extracted["identity_fields"] = identity

    # A few role phrases are unambiguous and security-relevant. Recover only
    # the role when the model omitted it; never recover names, PII, intent, or
    # authorization from pattern matching.
    role_text = message.casefold()
    representative_phrase = re.search(r"\bcalling\s+(?:for|on behalf of)\b", role_text)
    policyholder_phrase = re.search(
        r"\b(?:i am|i'm)\s+(?:the\s+)?policyholder\b", role_text
    )
    if representative_phrase and not policyholder_phrase:
        extracted["caller_role"] = "representative"
        extracted["identity_ambiguous"] = False
    elif policyholder_phrase and not representative_phrase:
        extracted["caller_role"] = "policyholder"
        extracted["identity_ambiguous"] = False

    if extracted["caller_role"] == "representative":
        # Explicit first-person representative fields never become
        # policyholder identity. Keep this ownership guard narrow: it only
        # removes a value when the customer labels it as their own.
        own_field_patterns = {
            "id_last4": r"\bmy\s+(?:own\s+)?(?:ssn|social security)\b",
            "dob": r"\bmy\s+(?:own\s+)?(?:dob|date of birth)\b",
            "email": r"\bmy\s+(?:own\s+)?email\b",
            "phone": r"\bmy\s+(?:own\s+)?phone\b",
        }
        for field_name, pattern in own_field_patterns.items():
            if field_name in identity and re.search(pattern, role_text):
                identity.pop(field_name)
                clear_name = f"identity_fields.{field_name}"
                if clear_name not in extracted["clear_fields"]:
                    extracted["clear_fields"].append(clear_name)

    # identity_fields contains policyholder PII, not representative PII.
    for name, value in list(identity.items()):
        if name not in IDENTITY_FIELDS:
            raise ModelResponseError(f"Unsupported identity field: {name}")

        if not isinstance(value, str) or not value.strip():
            raise ModelResponseError(f"Invalid identity value: {name}")

    # Validate complete field values. Never recover omitted values from regex,
    # assistant history, or fixtures; a semantic omission must remain omitted.
    for name, value in identity.items():
        normalized = IdentityVerifier._normalize_value(name, value)
        if normalized is None:
            # When Python explicitly asked for DOB, normalize the current
            # message itself as a narrowly bounded recovery. This prevents a
            # malformed model timestamp from rejecting an otherwise clear
            # answer, without searching arbitrary text for identity values.
            if (
                name == "dob"
                and conversation.pending_field == "dob"
                and IdentityVerifier._normalize_value("dob", message) is not None
            ):
                identity[name] = IdentityVerifier._normalize_value("dob", message)
                continue
            raise ModelResponseError("Invalid identity field format.")
        identity[name] = normalized
    for name in ("identity_ambiguous", "identity_context_changed"):
        if type(extracted[name]) is not bool:
            raise ModelResponseError("Invalid identity-context decision.")
    if extracted["general_question"] not in (
        None,
        "verification_reason",
        "identity_options",
        "document_preparation",
    ):
        raise ModelResponseError("Invalid general guidance topic.")
    if extracted["conversation_action"] not in ("continue", "wrap_up"):
        raise ModelResponseError("Invalid conversation action.")

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
            if not isinstance(value, str) or value not in allowed_values:
                raise ModelResponseError(f"Invalid extraction value: {name}")

    scope = extracted["scope"]

    if not isinstance(scope, str) or scope not in ALLOWED_SCOPES:
        raise ModelResponseError("Invalid scope value.")

    year = extracted["year"]

    if year is not None:
        if type(year) is not int or not 1 <= year <= 9999:
            raise ModelResponseError("year must be a valid integer year.")

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
            if not isinstance(value, str) or not value.strip():
                raise ModelResponseError(f"Invalid extraction value: {name}")

    if type(extracted["human_requested"]) is not bool:
        raise ModelResponseError("human_requested must be a boolean.")

    clear_fields = extracted["clear_fields"]

    if not isinstance(clear_fields, list):
        raise ModelResponseError("clear_fields must be a list.")

    if "id_last4" in identity and re.search(
        r"(?<![0-9])[0-9]{3}[- ][0-9]{2}[- ][0-9]{4}(?![0-9])|(?<![0-9])[0-9]{9}(?![0-9])",
        message,
    ):
        identity.pop("id_last4")
        clear_fields.append("identity_fields.id_last4")
        extracted["clarification_question"] = (
            "Please provide only the SSN last four, not a full SSN."
        )

    normalized_clear_fields = []

    for name in clear_fields:
        if not isinstance(name, str) or name not in ALLOWED_CLEAR_FIELDS:
            raise ModelResponseError("Unsupported field retraction.")

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
