"""
LangGraph controller for the SOP-guided insurance support agent.

This module defines:

    GraphState: Information passed between workflow nodes.

    InsuranceSOPGraph: Controls phase order, identity verification,
        claim access, scope handling, and email-summary consent.
"""

from copy import deepcopy
from dataclasses import fields
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAIError

from agent.case_handler import CaseHandler
from agent.extraction import extract_customer_information
from agent.memory import remember_customer_information
from agent.post_process import PostProcessHandler

from agent.state import ConversationState, Phase
from agent.verification import IdentityVerifier
from services.data_service import InsuranceDataService
from services.llm_service import LLMService


class GraphState(TypedDict):
    """
    Information passed through the workflow.

    Attributes:
        conversation: Working copy of the conversation state.
        message: Current customer message.
        extracted: Validated information from the current message.
        response: Customer-facing response for this turn.
    """

    conversation: ConversationState
    message: str
    extracted: dict[str, Any]
    response: str


class InsuranceSOPGraph:
    """
    Control the insurance support workflow.

    Attributes:
        data_service: Service used to access insurance fixtures.
        llm_service: Service used to call the model.
        identity_verifier: Deterministic identity-verification service.
        case_handler: Proposes case selections and generates claim answers.
        graph: Compiled LangGraph workflow.
        post_process_handler: Interprets consent and generates summary previews.
    """

    def __init__(
        self,
        data_service: InsuranceDataService,
        llm_service: LLMService,
    ) -> None:
        """
        Initialize services and compile the workflow.

        Args:
            data_service: Service used to query insurance fixtures.
            llm_service: Service used to call the model.
        """

        self.data_service = data_service
        self.llm_service = llm_service
        self.identity_verifier = IdentityVerifier(data_service)
        self.case_handler = CaseHandler(llm_service)
        self.post_process_handler = PostProcessHandler(llm_service)
        self.graph = self._build_graph()

    def handle_message(
        self,
        conversation: ConversationState,
        message: str,
    ) -> str:
        """
        Process one message and commit successful state updates.

        Args:
            conversation: Persistent state for this customer session.
            message: Current customer message.

        Returns:
            Customer-facing response.
        """

        message = message.strip()

        if not message:
            return "Please enter a message."

        if conversation.email_consent is not None:
            return (
                "This demo conversation is complete. "
                "Start a new conversation for another request."
            )

        # Prevent failed requests from leaving partial state updates.
        working = deepcopy(conversation)

        try:
            result = self.graph.invoke(
                {
                    "conversation": working,
                    "message": message,
                    "extracted": {},
                    "response": "",
                }
            )
        except (OpenAIError, ValueError, RuntimeError):
            #return (
            #    "I’m sorry, I couldn’t complete that step. "
            #    "Please try again. Your previous progress has been kept."
            #)
            raise

        updated = result["conversation"]
        response = result["response"]

        # The extractor receives earlier turns only.
        # Save the current turn after processing finishes.
        updated.messages.extend(
            [
                {
                    "role": "user",
                    "content": message,
                },
                {
                    "role": "assistant",
                    "content": response,
                },
            ]
        )

        for item in fields(ConversationState):
            setattr(
                conversation,
                item.name,
                getattr(updated, item.name),
            )

        return response

    def _build_graph(self) -> Any:
        """Build the workflow with explicit phase routing."""

        workflow = StateGraph(GraphState)

        workflow.add_node("extract", self._extract_node)
        workflow.add_node("verify_id", self._verify_id_node)
        workflow.add_node(
            "resolve_intent",
            self._resolve_intent_node,
        )
        workflow.add_node(
            "process_case",
            self._process_case_node,
        )
        workflow.add_node(
            "post_process",
            self._post_process_node,
        )

        workflow.add_edge(START, "extract")

        routes = {
            "verify_id": "verify_id",
            "resolve_intent": "resolve_intent",
            "process_case": "process_case",
            "post_process": "post_process",
            "end": END,
        }

        for node_name in (
            "extract",
            "verify_id",
            "resolve_intent",
            "post_process",
        ):
            workflow.add_conditional_edges(
                node_name,
                self._route,
                routes,
            )

        # Processing produces an answer and a summary offer.
        # Consent or another question arrives on the next turn.
        workflow.add_edge("process_case", END)

        return workflow.compile()

    def _route(self, state: GraphState) -> str:
        """
        Choose the next node according to verification and phase.

        Args:
            state: Current graph state.

        Returns:
            Next node name, or end when a response is ready.
        """

        if state["response"]:
            return "end"

        conversation = state["conversation"]

        if not conversation.is_verified:
            conversation.phase = Phase.VERIFY_ID
            return "verify_id"

        phase_routes = {
            Phase.VERIFY_ID: "verify_id",
            Phase.RESOLVE_INTENT: "resolve_intent",
            Phase.PROCESS_CASE: "process_case",
            Phase.POST_PROCESS: "post_process",
        }

        return phase_routes[conversation.phase]

    def _extract_node(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """Extract information, update memory, and apply scope gates."""

        conversation = state["conversation"]

        extracted = extract_customer_information(
            conversation,
            state["message"],
            self.llm_service.ask_model,
        )

        changed_identity = any(
            conversation.identity_fields.get(name) != value
            for name, value in extracted["identity_fields"].items()
        )

        retracted_identity = any(
            name.startswith("identity_fields.")
            for name in extracted["clear_fields"]
        )

        if conversation.is_verified and (
            changed_identity or retracted_identity
        ):
            # Require fresh verification without carrying forward
            # the previous identity, claim selection, or history.
            conversation = ConversationState()

        remember_customer_information(
            conversation,
            extracted,
        )

        response = ""

        if conversation.escalation_required:
            response = (
                "This demo has recorded your request for human "
                "assistance, but it cannot connect a live representative. "
                "A representative would still need to verify your identity."
            )

        elif extracted["scope"] == "out_of_scope":
            conversation.out_of_scope_attempts += 1

            if conversation.out_of_scope_attempts >= 3:
                conversation.escalation_required = True

                response = (
                    "I can help with insurance-support questions only. "
                    "Would you like help from a human representative? "
                    "Live transfer is not connected in this demo."
                )
            else:
                response = (
                    "I can help with insurance claims and related support, "
                    "but I can’t answer unrelated questions here."
                )

        else:
            # Count consecutive unrelated attempts.
            conversation.out_of_scope_attempts = 0

        return {
            "conversation": conversation,
            "extracted": extracted,
            "response": response,
        }

    def _verify_id_node(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """Verify identity before claim lookup or disclosure."""

        conversation = state["conversation"]

        policyholder = (
            self.identity_verifier.find_verified_policyholder(
                conversation.identity_fields
            )
        )

        if policyholder is None:
            empathy = ""

            if conversation.emotion is not None:
                empathy = (
                    "I understand this can be frustrating. "
                    "These checks help protect your private claim details. "
                )

            return {
                "response": empathy + (
                    "I haven’t been able to verify your identity yet. "
                    "I need three matching details from your full name, "
                    "date of birth, phone number including country code, "
                    "email, or SSN last four. You can use another listed "
                    "field if you prefer not to provide one, or ask "
                    "for human assistance."
                )
            }

        conversation.verified_party_id = policyholder["party_id"]
        conversation.phase = Phase.RESOLVE_INTENT

        # Continue within this turn using the remembered request.
        return {"conversation": conversation}

    def _resolve_intent_node(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """Resolve the request and validate the proposed claim selection."""

        conversation = state["conversation"]
        party_id = self._require_verified(conversation)

        if conversation.remembered_intent == "portal_support":
            return {
                "response": (
                    "Your identity is verified. I don’t have portal-account "
                    "tools or a documented troubleshooting procedure in "
                    "this demo. Would you like human assistance?"
                )
            }

        clarification = state["extracted"]["clarification_question"]

        if clarification:
            return {"response": clarification}

        if conversation.remembered_intent is None:
            return {
                "response": (
                    "Your identity is verified. "
                    "What would you like help with regarding your claim?"
                )
            }

        claims = self.data_service.find_claims_for_party(party_id)

        if not claims:
            return {
                "response": (
                    "I couldn’t find a claim linked to your verified record. "
                    "Would you like human assistance?"
                )
            }

        decision = self.case_handler.select_case(
            conversation=conversation,
            message=state["message"],
            claims=claims,
        )

        if decision["case_id"] is None:
            return {"response": decision["question"]}

        # The handler proposes a case; the controller checks ownership.
        claim = self.data_service.find_claim_for_party(
            party_id,
            decision["case_id"],
        )

        if claim is None:
            raise ValueError("The selected case is not authorized.")

        conversation.selected_case_id = claim["case_id"]
        conversation.phase = Phase.PROCESS_CASE

        return {"conversation": conversation}

    def _process_case_node(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """Check ownership, generate an answer, and offer a summary."""

        conversation = state["conversation"]

        # Recheck ownership before passing claim details to the handler.
        claim = self._get_selected_claim(conversation)

        answer = self.case_handler.generate_answer(
            conversation=conversation,
            message=state["message"],
            claim=claim,
            field_definitions=self.data_service.claim_schema,
            guidance=self.data_service.document_guidance,
        )

        conversation.phase = Phase.POST_PROCESS

        return {
            "conversation": conversation,
            "response": answer + (
                "\n\nWould you like an email summary of what we discussed "
                "and the next steps, or would you prefer to skip it? "
                "Email delivery is simulated in this demo. "
                "You can also ask another claim question."
            ),
        }

    def _post_process_node(
        self,
        state: GraphState,
    ) ->    dict[str, Any]:
        """Handle email consent or reopen the claim discussion."""

        conversation = state["conversation"]
        self._require_verified(conversation)

        action = self.post_process_handler.interpret_consent(
        conversation=conversation,
        message=state["message"],
        )

        if action == "followup":
            # Re-resolve the case because the customer may have supplied
            # a different request or corrected earlier case hints.
            conversation.selected_case_id = None
            conversation.phase = Phase.RESOLVE_INTENT

            return {"conversation": conversation}

        if action == "skip":
            conversation.email_consent = False

            return {
                "conversation": conversation,
                "response": (
                    "Understood. We’ll skip the email summary. "
                    "Thank you for contacting insurance support."
                ),
            }

        if action == "unclear":
            return {
                "response": (
                    "Would you like the email summary, "
                    "or should we skip it?"
                )
            }

        if action != "send":
            raise ValueError("Invalid email-consent decision.")

        # Ownership is checked before claim data reaches the handler.
        claim = self._get_selected_claim(conversation)

        body = self.post_process_handler.generate_summary(
            conversation=conversation,
            claim=claim,
            guidance=self.data_service.document_guidance,
        )

        # Record consent separately from email delivery.
        conversation.email_consent = True

        return {
            "conversation": conversation,
            "response": (
                "You chose the email summary. Here is its preview. "
                "No email was actually sent because delivery is simulated."
                f"\n\n{body}"
            ),
        }
    def _require_verified(
        self,
        conversation: ConversationState,
    ) -> str:
        """
        Require a verified policyholder ID before accessing claims.

        Args:
            conversation: Current conversation state.

        Returns:
            Verified policyholder ID.

        Raises:
            ValueError: If identity has not been verified.
        """

        party_id = conversation.verified_party_id

        if party_id is None:
            raise ValueError("Identity verification is required.")

        return party_id

    def _get_selected_claim(
        self,
        conversation: ConversationState,
    ) -> dict[str, Any]:
        """
        Recheck ownership whenever the selected claim is accessed.

        Args:
            conversation: Current conversation state.

        Returns:
            Selected claim belonging to the verified policyholder.

        Raises:
            ValueError: If no authorized claim is selected.
        """

        party_id = self._require_verified(conversation)
        case_id = conversation.selected_case_id

        if case_id is None:
            raise ValueError("No claim has been selected.")

        claim = self.data_service.find_claim_for_party(
            party_id,
            case_id,
        )

        if claim is None:
            raise ValueError("The selected claim is not authorized.")

        return claim