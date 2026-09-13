"""
Utilities for updating conversation memory.

This module defines:

    remember_customer_information: Merges validated customer information
        into conversation state without advancing the SOP workflow.
"""

from typing import Any

from agent.state import ConversationState


MEMORY_FIELDS = {
    "intent": "remembered_intent",
    "case_type": "remembered_case_type",
    "month": "remembered_month",
    "year": "remembered_year",
    "reported_status": "remembered_status_hint",
    "case_id": "remembered_case_id",
}


def remember_customer_information(
    conversation: ConversationState,
    extracted: dict[str, Any],
) -> None:
    """
    Merge validated extraction into conversation memory.

    Missing information preserves earlier values. Explicit corrections
    replace earlier values, while clear_fields removes retracted values.

    Args:
        conversation: Conversation state to update.
        extracted: Validated output from extract_customer_information.

    Raises:
        ValueError: If identity information changes after verification.
            The controller must handle re-verification before applying
            the update.
    """

    identity_updates = extracted["identity_fields"]
    clear_fields = extracted["clear_fields"]

    identity_changed = any(
        conversation.identity_fields.get(name) != value
        for name, value in identity_updates.items()
    )

    identity_retracted = any(
        name.startswith("identity_fields.")
        for name in clear_fields
    )

    # Check before making any changes to conversation memory.
    if conversation.is_verified and (
        identity_changed or identity_retracted
    ):
        raise ValueError(
            "Identity changes after verification require "
            "controller handling."
        )

    # Remove values explicitly retracted by the customer.
    for name in clear_fields:
        if name.startswith("identity_fields."):
            field_name = name.split(".", 1)[1]
            conversation.identity_fields.pop(field_name, None)
        else:
            setattr(conversation, MEMORY_FIELDS[name], None)

    # Add newly supplied identity fields or apply corrections.
    conversation.identity_fields.update(identity_updates)

    # Preserve remembered information when extraction returns None.
    for source, destination in MEMORY_FIELDS.items():
        value = extracted[source]

        if value is not None:
            setattr(conversation, destination, value)

    # Emotion describes the current turn.
    conversation.emotion = extracted["emotion"]

    # A human request remains active until the controller handles it.
    if extracted["human_requested"]:
        conversation.escalation_required = True