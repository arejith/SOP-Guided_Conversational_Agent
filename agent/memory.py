"""
Utilities for updating conversation memory.

This module defines:

    remember_customer_information: Merges validated customer information
        into conversation state without authorizing claim access.
"""

from typing import Any

from agent.state import CallerRole, ConversationState


MEMORY_FIELDS = {
    "intent": "remembered_intent",
    "case_type": "remembered_case_type",
    "month": "remembered_month",
    "year": "remembered_year",
    "reported_status": "remembered_status_hint",
    "case_id": "remembered_case_id",
}

REPRESENTATIVE_FIELDS = {
    "representative_name": "representative_name",
    "representative_relationship": "representative_relationship",
}

CALLER_FIELDS = {
    "caller_role",
    *REPRESENTATIVE_FIELDS,
}


def remember_customer_information(
    conversation: ConversationState,
    extracted: dict[str, Any],
) -> None:
    """
    Merge validated extraction into conversation memory.

    Missing information preserves earlier values. Explicit corrections
    replace earlier values, while clear_fields removes retracted values.

    Identity or caller-context changes after verification must first be
    handled by the controller. This function does not approve requests
    or advance workflow phases.

    Args:
        conversation: Conversation state to update.
        extracted: Validated output from extract_customer_information.

    Raises:
        ValueError: If identity or caller context changes after verification
            or after an authorization request has been created.
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

    # Convert the extracted role into the enum used by ConversationState.
    supplied_role = extracted["caller_role"]

    new_role = (
        CallerRole(supplied_role)
        if supplied_role is not None
        else None
    )

    role_changed = (
        new_role is not None
        and new_role != conversation.caller_role
    )

    representative_changed = any(
        extracted[source] is not None
        and extracted[source] != getattr(conversation, destination)
        for source, destination in REPRESENTATIVE_FIELDS.items()
    )

    caller_context_retracted = any(
        name in CALLER_FIELDS
        for name in clear_fields
    )

    security_context_changed = (
        identity_changed
        or identity_retracted
        or role_changed
        or representative_changed
        or caller_context_retracted
    )

    protected_context_exists = (
        conversation.is_verified
        or conversation.authorization_request_id is not None
    )

    # Check before changing any state. The controller must explicitly
    # invalidate verification and authorization when necessary.
    if protected_context_exists and security_context_changed:
        raise ValueError(
            "Identity or caller-context changes require "
            "controller handling before memory updates."
        )

    # Remove explicitly retracted information.
    for name in clear_fields:
        if name.startswith("identity_fields."):
            field_name = name.split(".", 1)[1]
            conversation.identity_fields.pop(field_name, None)

        elif name == "caller_role":
            conversation.caller_role = CallerRole.UNKNOWN

        elif name in REPRESENTATIVE_FIELDS:
            setattr(
                conversation,
                REPRESENTATIVE_FIELDS[name],
                None,
            )

        else:
            setattr(
                conversation,
                MEMORY_FIELDS[name],
                None,
            )

    # identity_fields contains policyholder information only.
    conversation.identity_fields.update(identity_updates)

    if new_role is not None:
        conversation.caller_role = new_role

    for source, destination in REPRESENTATIVE_FIELDS.items():
        value = extracted[source]

        if value is not None:
            setattr(conversation, destination, value)

    # Do not retain a previous representative's details after the
    # caller's role is cleared or changed to policyholder.
    if (
        role_changed
        or "caller_role" in clear_fields
    ) and conversation.caller_role != CallerRole.REPRESENTATIVE:
        conversation.representative_name = None
        conversation.representative_relationship = None

    # Preserve earlier claim hints when the current message omits them.
    for source, destination in MEMORY_FIELDS.items():
        value = extracted[source]

        if value is not None:
            setattr(conversation, destination, value)

    # Emotion describes the current turn.
    conversation.emotion = extracted["emotion"]

    # A human-assistance request remains active until handled.
    if extracted["human_requested"]:
        conversation.escalation_required = True