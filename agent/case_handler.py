"""
Case interpretation and response generation for insurance support.

This module defines:

    CaseHandler: Proposes claim selections and generates answers using
        claim data supplied by the SOP controller.
"""

from datetime import date
from typing import Any

from agent.prompts import (
    CASE_SELECTION_PROMPT,
    CLAIM_ANSWER_PROMPT,
)
from agent.state import ConversationState
from services.errors import ModelResponseError
from services.llm_service import LLMService


def applicable_guidance(
    claim: dict[str, Any], guidance: dict[str, Any]
) -> dict[str, Any]:
    """Limit answer and summary context to guidance relevant to this record."""

    documents = claim.get("documents_needed", [])
    aliases = {"office note": "treating provider office note"}
    document_keys = {aliases.get(name, name) for name in documents}
    selected = {
        "default_guidance": guidance.get("default_guidance", {}),
        "case_type_guidance": guidance.get("case_type_guidance", {}).get(
            claim.get("case_type"), {}
        ),
        "document_guidance": {
            name: value
            for name, value in guidance.get("document_guidance", {}).items()
            if name in document_keys
        },
    }
    if documents:
        selected["document_alternative_guidance"] = {
            name: value
            for name, value in guidance.get("document_alternative_guidance", {}).items()
            if name == "default" or name in document_keys
        }
        selected["claim_followup_guidance"] = [
            item
            for item in guidance.get("claim_followup_guidance", [])
            # A generic relative timing instruction must not compete with a
            # recorded fixed appeal deadline.
            if not (
                claim.get("appeal_deadline")
                and item.get("topic") == "submission_timing"
            )
        ]
        selected["claim_followup_settings"] = guidance.get(
            "claim_followup_settings", {}
        )
    return selected


class CaseHandler:
    """
    Interpret claim requests and generate grounded responses.

    Attributes:
        llm_service: Service used to call the model and validate responses.
    """

    def __init__(self, llm_service: LLMService) -> None:
        """
        Initialize the case handler.

        Args:
            llm_service: Service used to access the model.
        """

        self.llm_service = llm_service

    def select_case(
        self,
        conversation: ConversationState,
        message: str,
        claims: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Propose a case selection or a clarification question.

        Args:
            conversation: Conversation context and remembered hints.
            message: Current customer message.
            claims: Candidate claims belonging to the verified policyholder,
                supplied by the controller.

        Returns:
            A proposed case_id and question. Exactly one is non-null.

        Raises:
            ValueError: If the model returns an invalid decision.
        """

        decision = self.llm_service.ask_json(
            CASE_SELECTION_PROMPT,
            {
                "access_context": {"claim_access_authorized": True},
                "request": message,
                "recent_messages": conversation.messages[-8:],
                "intent": conversation.remembered_intent,
                "hints": {
                    "case_id": conversation.remembered_case_id,
                    "case_type": conversation.remembered_case_type,
                    "month": conversation.remembered_month,
                    "year": conversation.remembered_year,
                    "reported_status": (conversation.remembered_status_hint),
                },
                "candidates": [
                    {
                        key: claim.get(key)
                        for key in (
                            "case_id",
                            "case_type",
                            "created_at",
                            "status",
                        )
                    }
                    for claim in claims
                ],
            },
        )

        if set(decision) != {"case_id", "question"}:
            raise ModelResponseError("Invalid case-selection response.")

        case_id = decision["case_id"]

        if case_id is None:
            return {
                "case_id": None,
                "question": self.llm_service.required_text(
                    decision,
                    "question",
                ),
            }

        if not isinstance(case_id, str) or not case_id.strip():
            raise ModelResponseError("Invalid selected case ID.")

        if decision["question"] is not None:
            raise ModelResponseError("Ambiguous case-selection response.")

        # This is only a proposal. The controller checks ownership.
        return {
            "case_id": case_id.strip(),
            "question": None,
        }

    def generate_answer(
        self,
        conversation: ConversationState,
        message: str,
        claim: dict[str, Any],
        field_definitions: dict[str, Any],
        guidance: dict[str, Any],
    ) -> str:
        """
        Generate an answer using the supplied claim and guidance.

        Args:
            conversation: Conversation context and remembered intent.
            message: Current customer message.
            claim: Claim whose ownership was checked by the controller.
            field_definitions: Definitions of fields in the claim record.
            guidance: Document and follow-up guidance from fixtures.

        Returns:
            Customer-facing answer without an email-summary offer.

        Raises:
            ValueError: If the model returns an invalid answer structure.
        """

        result = self.llm_service.ask_json(
            CLAIM_ANSWER_PROMPT,
            {
                "access_context": {"claim_access_authorized": True},
                "today": date.today().isoformat(),
                "request": message,
                "intent": conversation.remembered_intent,
                "emotion": conversation.emotion,
                "recent_messages": conversation.messages[-8:],
                "claim": claim,
                "field_definitions": field_definitions,
                "guidance": applicable_guidance(claim, guidance),
            },
        )

        if set(result) != {"answer"}:
            raise ModelResponseError("Invalid claim-answer response.")

        return self.llm_service.required_text(result, "answer")
