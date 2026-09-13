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

from agent.state import ConversationState, Phase, CallerRole
from agent.verification import IdentityVerifier
from services.data_service import InsuranceDataService
from services.llm_service import LLMService
from services.authorization_service import (
    RepresentativeAuthorizationService,
)


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

        self.authorization_service = RepresentativeAuthorizationService(
            data_service=data_service,
            scenario="default",
        )

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
        Route according to identity, authorization, and workflow phase.

        Args:
            state: Current graph state.

        Returns:
            Next node name, or end when a response is ready.
        """

        if state["response"]:
            return "end"

        conversation = state["conversation"]

        if (
            not conversation.is_verified
            or conversation.caller_role == CallerRole.UNKNOWN
        ):
            conversation.phase = Phase.VERIFY_ID
            return "verify_id"

        if conversation.caller_role == CallerRole.REPRESENTATIVE:
            request_id = conversation.authorization_request_id
            representative_name = conversation.representative_name
            relationship = conversation.representative_relationship

            if not request_id or not representative_name or not relationship:
                conversation.phase = Phase.VERIFY_ID
                return "verify_id"

            approved = self.authorization_service.is_approved(
                request_id=request_id,
                verified_party_id=conversation.verified_party_id,
                representative_name=representative_name,
                relationship=relationship,
            )

            if not approved:
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
        """Extract information and safely update the caller context."""

        conversation = state["conversation"]

        extracted = extract_customer_information(
            conversation,
            state["message"],
            self.llm_service.ask_model,
        )

        clear_fields = extracted["clear_fields"]

        identity_changed = any(
            conversation.identity_fields.get(name) != value
            for name, value in extracted["identity_fields"].items()
        )

        identity_retracted = any(
            name.startswith("identity_fields.")
            for name in clear_fields
        )

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

        representative_fields = (
            "representative_name",
            "representative_relationship",
        )

        representative_changed = any(
            extracted[name] is not None
            and extracted[name] != getattr(conversation, name)
            for name in representative_fields
        )

        caller_context_retracted = any(
            name in {
                "caller_role",
                "representative_name",
                "representative_relationship",
            }
            for name in clear_fields
        )

        protected_context_exists = (
            conversation.is_verified
            or conversation.authorization_request_id is not None
        )

        security_context_changed = (
            identity_changed
            or identity_retracted
            or role_changed
            or representative_changed
            or caller_context_retracted
        )

        # A caller switch can also happen before verification.
        established_role_changed = (
            conversation.caller_role != CallerRole.UNKNOWN
            and (
                role_changed
                or "caller_role" in clear_fields
            )
        )

        established_representative_changed = any(
            getattr(conversation, name) is not None
            and (
                name in clear_fields
                or (
                    extracted[name] is not None
                    and extracted[name] != getattr(conversation, name)
                )
            )
            for name in representative_fields
        )

        reset_context = (
            protected_context_exists and security_context_changed
        ) or established_role_changed or established_representative_changed

        if reset_context:
            # Drop access credentials, previous identity, claim selection,
            # and history. Collect the new caller context explicitly.
            previous_attempts = conversation.out_of_scope_attempts
            previous_escalation = conversation.escalation_required

            conversation = ConversationState(
                out_of_scope_attempts=previous_attempts,
                escalation_required=previous_escalation,
            )

        remember_customer_information(
            conversation,
            extracted,
        )

        response = ""

        if conversation.escalation_required:
            response = (
                "This demo has recorded the need for human assistance, "
                "but it cannot connect a live representative. "
                "Identity and any required authorization checks "
                "would still apply."
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
        """Complete identity and representative-authorization checks."""

        conversation = state["conversation"]
        conversation.phase = Phase.VERIFY_ID

        empathy = {
            "frustration": (
                "I understand these extra steps can be frustrating. "
            ),
            "anger": (
                "I hear that you’re upset, and I want to help. "
            ),
            "anxiety": (
                "I understand this may feel worrying. "
                "We can take it one step at a time. "
            ),
            "confusion": (
                "I’m happy to explain what we need. "
            ),
            "refusal": (
                "You can choose not to continue with these checks. "
            ),
        }.get(conversation.emotion, "")

        if conversation.caller_role == CallerRole.UNKNOWN:
            return {
                "response": empathy + (
                    "Are you the policyholder, or are you calling "
                    "on behalf of someone else?"
                )
            }

        is_representative = (
            conversation.caller_role == CallerRole.REPRESENTATIVE
        )

        if is_representative:
            missing_details = []

            if not conversation.representative_name:
                missing_details.append("your full name")

            if not conversation.representative_relationship:
                missing_details.append(
                    "your relationship to the policyholder"
                )

            if missing_details:
                return {
                    "response": empathy + (
                        "Please provide "
                        + " and ".join(missing_details)
                        + ". These are your details; the identity "
                        "verification fields must belong to the policyholder."
                    )
                }

        # A refusal must not trigger an authorization-status advance.
        if conversation.emotion == "refusal":
            return {
                "response": empathy + (
                    "I can’t disclose claim details without the required "
                    "identity and authorization checks. You can use another "
                    "permitted identity field, ask the policyholder to "
                    "contact support directly, or request human assistance."
                )
            }

        if not conversation.is_verified:
            policyholder = (
                self.identity_verifier.find_verified_policyholder(
                    conversation.identity_fields
                )
            )

            if policyholder is None:
                subject = (
                    "the policyholder’s"
                    if is_representative
                    else "your"
                )

                return {
                    "response": empathy + (
                        "To protect private claim information, I need "
                        f"three matching details from {subject} full name, "
                        "date of birth, phone number including country code, "
                        "email, or SSN last four. "
                        "You can choose which three to provide."
                    )
                }

            conversation.verified_party_id = policyholder["party_id"]

        if not is_representative:
            self._require_verified(conversation)
            conversation.phase = Phase.RESOLVE_INTENT

            return {"conversation": conversation}

        party_id = conversation.verified_party_id

        if party_id is None:
            raise ValueError("Policyholder verification is required.")

        representative_name = conversation.representative_name
        relationship = conversation.representative_relationship

        if not representative_name or not relationship:
            raise ValueError("Representative details are required.")

        if conversation.authorization_request_id is None:
            # A listed relationship is a prerequisite, not approval.
            representative = self.data_service.find_representative(
                party_id=party_id,
                representative_name=representative_name,
                relationship=relationship,
            )

            if representative is None:
                conversation.escalation_required = True

                return {
                    "conversation": conversation,
                    "response": empathy + (
                        "I couldn’t establish your representative relationship "
                        "from the information provided. I can’t disclose claim "
                        "details. Human assistance is needed to review access; "
                        "live transfer is not connected in this demo."
                    ),
                }

            conversation.authorization_request_id = (
                self.authorization_service.begin_request(
                    verified_party_id=party_id,
                    representative_name=representative_name,
                    relationship=relationship,
                )
            )

        status = self.authorization_service.check_status(
            request_id=conversation.authorization_request_id,
            verified_party_id=party_id,
        )

        if status == "pending":
            return {
                "conversation": conversation,
                "response": empathy + (
                    "The policyholder identity details have been verified, "
                    "but authorization for you to access the claim is pending. "
                    "This demo simulates policyholder approval; it has not "
                    "contacted the policyholder. You can ask me to check "
                    "authorization again, or request human assistance. "
                    "I can’t share claim details while approval is pending."
                ),
            }

        if status == "approved":
            # Recheck the complete binding before advancing.
            self._require_verified(conversation)
            conversation.phase = Phase.RESOLVE_INTENT

            return {"conversation": conversation}

        if status in {"denied", "timeout"}:
            conversation.escalation_required = True

            explanation = (
                "The simulated authorization request was denied."
                if status == "denied"
                else "The simulated authorization request timed out."
            )

            return {
                "conversation": conversation,
                "response": empathy + (
                    explanation
                    + " I can’t disclose claim details. "
                    "The policyholder can contact support directly, "
                    "or a human representative can review authorization. "
                    "Live transfer is not connected in this demo."
                ),
            }

        raise ValueError("Unexpected authorization status.")

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
    ) -> dict[str, Any]:
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
        Require identity verification and any necessary authorization.

        Args:
            conversation: Current conversation state.

        Returns:
            Policyholder ID permitted for claim access.

        Raises:
            ValueError: If identity, caller role, or representative
                authorization has not been established.
        """

        party_id = conversation.verified_party_id

        if party_id is None:
            raise ValueError("Identity verification is required.")

        if conversation.caller_role == CallerRole.POLICYHOLDER:
            return party_id

        if conversation.caller_role != CallerRole.REPRESENTATIVE:
            raise ValueError("The caller's role must be established.")

        request_id = conversation.authorization_request_id
        representative_name = conversation.representative_name
        relationship = conversation.representative_relationship

        if not request_id or not representative_name or not relationship:
            raise ValueError("Representative authorization is required.")

        approved = self.authorization_service.is_approved(
            request_id=request_id,
            verified_party_id=party_id,
            representative_name=representative_name,
            relationship=relationship,
        )

        if not approved:
            raise ValueError(
                "Representative authorization has not been approved."
            )

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