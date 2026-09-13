"""
Post-processing support for the insurance agent.

This module defines:

    PostProcessHandler: Interprets email-summary choices and generates
        summary previews without sending emails or changing state.
"""

from datetime import date
from typing import Any

from agent.prompts import (
    EMAIL_CONSENT_PROMPT,
    EMAIL_SUMMARY_PROMPT,
)
from agent.state import ConversationState
from services.llm_service import LLMService


class PostProcessHandler:
    """
    Interpret email consent and generate conversation summaries.

    Attributes:
        llm_service: Service used to access the model.
    """

    def __init__(self, llm_service: LLMService) -> None:
        """
        Initialize the post-processing handler.

        Args:
            llm_service: Service used to call the model.
        """

        self.llm_service = llm_service

    def interpret_consent(
        self,
        conversation: ConversationState,
        message: str,
    ) -> str:
        """
        Interpret the customer's response to the email-summary offer.

        Args:
            conversation: Conversation context containing the earlier offer.
            message: Current customer message.

        Returns:
            One of send, skip, followup, or unclear.

        Raises:
            ValueError: If the model returns an invalid decision.
        """

        decision = self.llm_service.ask_json(
            EMAIL_CONSENT_PROMPT,
            {
                "recent_messages": conversation.messages[-6:],
                "message": message,
            },
        )

        if set(decision) != {"action"}:
            raise ValueError("Invalid email-consent response.")

        action = decision["action"]

        allowed_actions = {
            "send",
            "skip",
            "followup",
            "unclear",
        }

        if not isinstance(action, str) or action not in allowed_actions:
            raise ValueError("Invalid email-consent decision.")

        return action

    def generate_summary(
        self,
        conversation: ConversationState,
        claim: dict[str, Any],
        guidance: dict[str, Any],
    ) -> str:
        """
        Generate an email-summary preview using supplied claim data.

        Args:
            conversation: Conversation to summarize.
            claim: Claim whose ownership was checked by the controller.
            guidance: Applicable guidance from the fixture data.

        Returns:
            Generated email body.

        Raises:
            ValueError: If the model returns an invalid summary.
        """

        result = self.llm_service.ask_json(
            EMAIL_SUMMARY_PROMPT,
            {
                "today": date.today().isoformat(),
                "conversation": conversation.messages,
                "claim": claim,
                "guidance": guidance,
            },
        )

        if set(result) != {"summary"}:
            raise ValueError("Invalid email-summary response.")

        return self.llm_service.required_text(
            result,
            "summary",
        )