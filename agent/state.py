"""
Conversation state for the SOP-guided insurance support agent.

This module defines:

    Phase: Required phases of the business workflow.

    CallerRole: Whether the caller is the policyholder or a representative.

    ConversationState: Identity information, conversation memory,
        authorization context, and workflow progress.
"""

from dataclasses import dataclass, field
from enum import Enum


class Phase(str, Enum):
    """The required phases of the insurance support workflow."""

    VERIFY_ID = "VERIFY_ID"
    RESOLVE_INTENT = "RESOLVE_INTENT"
    PROCESS_CASE = "PROCESS_CASE"
    POST_PROCESS = "POST_PROCESS"


class CallerRole(str, Enum):
    """The caller's stated role in the conversation."""

    UNKNOWN = "unknown"
    POLICYHOLDER = "policyholder"
    REPRESENTATIVE = "representative"


@dataclass
class ConversationState:
    """
    Store information and workflow progress for one conversation.

    Attributes:
        phase: Current SOP phase.
        caller_role: Whether the caller is the policyholder or a representative.
        identity_fields: Policyholder identity fields supplied by the caller.
        verified_party_id: Policyholder ID established by PII verification.
        representative_name: Name of the person acting for the policyholder.
        representative_relationship: Their stated relationship to the policyholder.
        authorization_request_id: Request ID created by the authorization service.
        remembered_intent: The customer's interpreted request.
        remembered_case_type: Case type mentioned by the customer.
        remembered_month: Month mentioned by the customer.
        remembered_year: Year explicitly provided or clarified.
        remembered_status_hint: Claim status reported by the customer.
        remembered_case_id: Case ID mentioned by the customer.
        selected_case_id: Case selected after access and ownership checks.
        messages: User and assistant messages for conversation context.
        emotion: Emotion detected in the current customer message.
        out_of_scope_attempts: Number of consecutive unrelated-question attempts.
        email_consent: True to send, False to skip, or None if undecided.
        escalation_required: Whether human assistance is needed.
        pending_question: Last focused question used to interpret short replies.
        pending_field: Field requested by that question, when known.
        verification_attempts: Count of unsuccessful verification turns.
        clarification_attempts: Consecutive semantic clarification requests.
        summary_delivery_status: not_requested, skipped, or preview_generated.
        summary_preview: Generated body; never evidence of actual delivery.
    """

    phase: Phase = Phase.VERIFY_ID

    caller_role: CallerRole = CallerRole.UNKNOWN

    identity_fields: dict[str, str] = field(default_factory=dict)
    verified_party_id: str | None = None

    representative_name: str | None = None
    representative_relationship: str | None = None
    authorization_request_id: str | None = None

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
    pending_question: str | None = None
    pending_field: str | None = None
    verification_attempts: int = 0
    clarification_attempts: int = 0
    summary_delivery_status: str = "not_requested"
    summary_preview: str | None = None

    @property
    def is_verified(self) -> bool:
        """
        Return whether policyholder PII verification has succeeded.

        This does not establish representative authorization.
        """

        return self.verified_party_id is not None
