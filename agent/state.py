"""
Conversation state for the SOP-guided insurance support agent.

This module defines:

    Phase: The four required phases of the business workflow.

    ConversationState: Identity information, conversation memory,
        and workflow progress for one customer session.
"""

from dataclasses import dataclass, field
from enum import Enum


class Phase(str, Enum):
    """The required phases of the insurance support workflow."""

    VERIFY_ID = "VERIFY_ID"
    RESOLVE_INTENT = "RESOLVE_INTENT"
    PROCESS_CASE = "PROCESS_CASE"
    POST_PROCESS = "POST_PROCESS"


@dataclass
class ConversationState:
    """
    Store information and workflow progress for one conversation.

    Attributes:
        phase: The agent's current SOP phase.
        identity_fields: Identity fields supplied by the customer.
        verified_party_id: Policyholder ID after successful verification.
        remembered_intent: The customer's interpreted request.
        remembered_case_type: Case type mentioned by the customer.
        remembered_month: Month mentioned by the customer.
        remembered_year: Year explicitly provided or clarified.
        remembered_status_hint: Claim status reported by the customer,
            which must be checked against the claim record.
        remembered_case_id: Case ID mentioned by the customer.
        selected_case_id: Case selected after verification and an
            ownership check.
        messages: User and assistant messages for conversation context.
        emotion: Customer emotion detected from the conversation.
        out_of_scope_attempts: Number of unrelated-question attempts.
        email_consent: True to send, False to skip, or None if undecided.
        escalation_required: Whether human assistance is needed.
    """

    phase: Phase = Phase.VERIFY_ID
    identity_fields: dict[str, str] = field(default_factory=dict)
    verified_party_id: str | None = None

    remembered_intent: str | None = None
    remembered_case_type: str | None = None
    remembered_month: str | None = None
    remembered_year: int | None = None
    remembered_status_hint: str | None = None
    remembered_case_id: str | None = None

    selected_case_id: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list)

    emotion: str | None = None
    out_of_scope_attempts: int = 0
    email_consent: bool | None = None
    escalation_required: bool = False

    @property
    def is_verified(self) -> bool:
        """Return whether a verified policyholder ID has been assigned."""
        return self.verified_party_id is not None