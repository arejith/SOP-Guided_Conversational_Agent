"""
LangGraph controller for the SOP-guided insurance support agent.

This module defines:

    GraphState: Information passed between workflow nodes.

    InsuranceSOPGraph: Controls phase order, identity verification,
        claim access, scope handling, and email-summary consent.
"""

import logging
from copy import deepcopy
from dataclasses import fields
from datetime import datetime
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAIError

from agent.case_handler import CaseHandler
from agent.extraction import extract_customer_information, extraction_schema
from agent.memory import remember_customer_information
from agent.post_process import PostProcessHandler
from agent.state import CallerRole, ConversationState, Phase
from agent.verification import IdentityVerifier
from services.authorization_service import (
    RepresentativeAuthorizationService,
)
from services.data_service import InsuranceDataService
from services.errors import ModelResponseError
from services.llm_service import LLMService

logger = logging.getLogger(__name__)


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
        authorization_before = deepcopy(self.authorization_service.requests)
        self._security_invalidated = False

        try:
            result = self.graph.invoke(
                {
                    "conversation": working,
                    "message": message,
                    "extracted": {},
                    "response": "",
                }
            )
        except Exception as error:
            # Simulator mutations participate in the same turn transaction.
            self.authorization_service.requests = authorization_before
            if self._security_invalidated:
                # A failed answer must not restore a prior person's access.
                clean = ConversationState()
                for item in fields(ConversationState):
                    setattr(conversation, item.name, getattr(clean, item.name))
            if not isinstance(error, (OpenAIError, ModelResponseError)):
                raise
            # Log type and phase only: exceptions can contain credentials or PII.
            logger.warning(
                "Model step failed: phase=%s error_type=%s",
                working.phase.value,
                type(error).__name__,
            )
            return "I’m sorry, I couldn’t complete that step. Please try again. " + (
                "We need to verify the current caller before continuing."
                if self._security_invalidated
                else "Your previous progress has been kept."
            )

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
            "process_case",
        ):
            workflow.add_conditional_edges(
                node_name,
                self._route,
                routes,
            )

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
        normalized_message = " ".join(state["message"].casefold().split())
        explicit_summary_choice = (
            conversation.phase == Phase.POST_PROCESS
            and normalized_message.strip(" .,!?")
            in {
                "yes",
                "yes send",
                "yes send it",
                "yes please send",
                "yes please send it",
                "yes send the email",
                "yes send the email summary",
                "send the email summary",
                "please send the email summary",
                "no",
                "no thanks",
                "skip",
                "skip it",
                "skip the email",
                "skip the email summary",
                "do not send it",
                "don't send it",
            }
        )
        if explicit_summary_choice:
            extracted = {
                "identity_fields": {},
                "caller_role": None,
                "representative_name": None,
                "representative_relationship": None,
                "intent": None,
                "case_type": None,
                "month": None,
                "year": None,
                "reported_status": None,
                "case_id": None,
                "emotion": None,
                "scope": "in_scope",
                "human_requested": False,
                "clarification_question": None,
                "clear_fields": [],
                "identity_ambiguous": False,
                "identity_context_changed": False,
                "general_question": None,
                "conversation_action": "wrap_up",
            }
        else:
            extracted = extract_customer_information(
                conversation,
                state["message"],
                lambda prompt: self.llm_service.ask_model(
                    prompt, schema=extraction_schema()
                ),
            )

        clear_fields = extracted["clear_fields"]
        normalize = IdentityVerifier._normalize_value
        identity_changed = any(
            normalize(name, conversation.identity_fields.get(name)) != value
            for name, value in extracted["identity_fields"].items()
        )
        identity_retracted = any(
            name.startswith("identity_fields.") for name in clear_fields
        )
        supplied_role = extracted["caller_role"]
        role_changed = (
            supplied_role is not None
            and supplied_role != conversation.caller_role.value
            and conversation.caller_role != CallerRole.UNKNOWN
        ) or "caller_role" in clear_fields
        representative_changed = any(
            getattr(conversation, name) is not None
            and (
                name in clear_fields
                or (
                    extracted[name] is not None
                    and " ".join(extracted[name].casefold().split())
                    != " ".join(getattr(conversation, name).casefold().split())
                )
            )
            for name in ("representative_name", "representative_relationship")
        )
        new_person = (
            role_changed
            or representative_changed
            or (
                extracted["identity_context_changed"]
                and any(
                    name in extracted["identity_fields"]
                    for name in ("name", "policy_number")
                )
            )
            or extracted["identity_ambiguous"]
            or any(
                name in conversation.identity_fields
                and name in extracted["identity_fields"]
                and normalize(name, conversation.identity_fields[name])
                != extracted["identity_fields"][name]
                for name in ("name", "policy_number")
            )
        )
        if new_person:
            previous_clarifications = conversation.clarification_attempts
            self._security_invalidated = True
            if conversation.authorization_request_id:
                self.authorization_service.requests.pop(
                    conversation.authorization_request_id, None
                )
            conversation = ConversationState()
            conversation.clarification_attempts = previous_clarifications
            if extracted["identity_ambiguous"]:
                extracted["identity_fields"] = {}
                extracted["caller_role"] = None
                extracted["representative_name"] = None
                extracted["representative_relationship"] = None
                extracted["clarification_question"] = extracted[
                    "clarification_question"
                ] or ("Who is speaking, and who is the policyholder?")
        elif identity_changed or identity_retracted:
            if conversation.is_verified or conversation.authorization_request_id:
                self._security_invalidated = True
                if conversation.authorization_request_id:
                    self.authorization_service.requests.pop(
                        conversation.authorization_request_id, None
                    )
                conversation.verified_party_id = None
                conversation.authorization_request_id = None
                conversation.selected_case_id = None
                conversation.phase = Phase.VERIFY_ID
                conversation.messages = []
                conversation.email_consent = None
                conversation.summary_preview = None
                conversation.summary_delivery_status = "not_requested"

        # Harmless formatting changes do not invalidate existing access.
        for name, value in list(conversation.identity_fields.items()):
            normalized = normalize(name, value)
            if normalized is not None:
                conversation.identity_fields[name] = normalized
        for name in ("representative_name", "representative_relationship"):
            if extracted[name] and getattr(conversation, name):
                if " ".join(extracted[name].casefold().split()) == " ".join(
                    getattr(conversation, name).casefold().split()
                ):
                    extracted[name] = getattr(conversation, name)

        hint_changed = any(
            (
                extracted[source] is not None
                and extracted[source] != getattr(conversation, target)
            )
            or source in clear_fields
            for source, target in (
                ("case_id", "remembered_case_id"),
                ("case_type", "remembered_case_type"),
                ("month", "remembered_month"),
                ("year", "remembered_year"),
            )
        )
        remember_customer_information(conversation, extracted)
        if hint_changed and conversation.selected_case_id:
            conversation.selected_case_id = None
            conversation.phase = Phase.RESOLVE_INTENT

        response = ""
        if extracted["human_requested"]:
            response = (
                "This demo has recorded your request for human assistance, but it "
                "cannot connect a live representative. Identity and any required "
                "authorization checks would still apply. You may also continue here."
            )
        elif extracted["scope"] == "out_of_scope":
            conversation.out_of_scope_attempts += 1
            response = "I can help with insurance claims and related support, but I can’t answer unrelated questions here."
            if conversation.out_of_scope_attempts >= 3:
                response += " Would you like human assistance? Live transfer is not connected in this demo."
        else:
            conversation.out_of_scope_attempts = 0
            conversation.escalation_required = False
            if extracted["clarification_question"]:
                conversation.clarification_attempts += 1
                conversation.pending_question = extracted["clarification_question"]
                conversation.pending_field = (
                    "caller_role" if extracted["identity_ambiguous"] else None
                )
                response = conversation.pending_question
                if conversation.clarification_attempts >= 3:
                    response += " If this is getting difficult, you can request human assistance. Live transfer is not connected in this demo."
            else:
                conversation.clarification_attempts = 0
                if not extracted["general_question"]:
                    conversation.pending_question = None
                    conversation.pending_field = None

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
            "frustration": ("I understand these extra steps can be frustrating. "),
            "anger": ("I hear that you’re upset, and I want to help. "),
            "anxiety": (
                "I understand this may feel worrying. "
                "We can take it one step at a time. "
            ),
            "confusion": ("I’m happy to explain what we need. "),
            "refusal": ("You can choose not to continue with these checks. "),
        }.get(conversation.emotion, "")

        general = state["extracted"]["general_question"]
        if general:
            public_answers = {
                "verification_reason": "Verification protects private claim information. We need three matching permitted identity details before discussing a claim; representatives also need authorization.",
                "identity_options": "You may use full name, date of birth, phone with country code, email, or SSN last four. Choose three. A policy number can help locate the policy but does not count. Please do not provide a full SSN.",
                "document_preparation": self.data_service.document_guidance[
                    "default_guidance"
                ]["en"],
            }
            return {
                "response": empathy
                + public_answers[general]
                + " This is general guidance; claim-specific help requires verification."
            }

        if conversation.caller_role == CallerRole.UNKNOWN:
            conversation.pending_field = "caller_role"
            conversation.pending_question = (
                "Are you the policyholder, or calling for someone else?"
            )
            return {
                "response": empathy
                + (
                    "Are you the policyholder, or are you calling "
                    "on behalf of someone else?"
                )
            }

        is_representative = conversation.caller_role == CallerRole.REPRESENTATIVE

        if is_representative:
            missing_details = []

            if not conversation.representative_name:
                missing_details.append("your full name")

            if not conversation.representative_relationship:
                missing_details.append("your relationship to the policyholder")

            if missing_details:
                conversation.pending_field = (
                    "representative_name"
                    if not conversation.representative_name
                    else "representative_relationship"
                )
                conversation.pending_question = (
                    "Please provide " + " and ".join(missing_details) + "."
                )
                return {
                    "response": empathy
                    + (
                        "Please provide "
                        + " and ".join(missing_details)
                        + ". These are your details; the identity "
                        "verification fields must belong to the policyholder."
                    )
                }

        # A refusal must not trigger an authorization-status advance.
        if conversation.emotion == "refusal":
            return {
                "response": empathy
                + (
                    "I can’t disclose claim details without the required "
                    "identity and authorization checks. You can use another "
                    "permitted identity field, ask the policyholder to "
                    "contact support directly, or request human assistance."
                )
            }

        if not conversation.is_verified:
            policyholder = self.identity_verifier.find_verified_policyholder(
                conversation.identity_fields
            )

            if policyholder is None:
                subject = "the policyholder’s" if is_representative else "your"

                conversation.verification_attempts += 1
                labels = {
                    "name": "full name",
                    "dob": "date of birth",
                    "phone": "phone number including country code",
                    "email": "email address",
                    "id_last4": "SSN last four",
                }
                available = [
                    name for name in labels if name not in conversation.identity_fields
                ]
                supplied_count = sum(
                    name in conversation.identity_fields for name in labels
                )
                target = available[0] if available else "name"
                conversation.pending_field = target
                question = f"Could you provide {subject} {labels[target]}?"
                if available[1:]:
                    question += (
                        " You can use "
                        + " or ".join(labels[name] for name in available[1:])
                        + " instead."
                    )
                if supplied_count < 3:
                    question = (
                        "To protect claim information, I need three permitted identity details. "
                        + question
                    )
                else:
                    question = (
                        "I couldn’t verify the information provided. Please check your details or try another permitted field. "
                        + question
                    )
                if conversation.verification_attempts >= 3:
                    question += " If you prefer, you can request human assistance; live transfer is not connected here."
                conversation.pending_question = question
                return {"response": empathy + question}

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
                    "response": empathy
                    + (
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
                "response": empathy
                + (
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
                "response": empathy
                + (
                    explanation + " I can’t disclose claim details. "
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

        # An authorization-only follow-up is a controller action, not a new
        # intent. Preserve the request captured before authorization completed.
        recheck_text = " ".join(state["message"].casefold().split())
        is_authorization_recheck = "authorization" in recheck_text and any(
            word in recheck_text for word in ("again", "check", "status", "approved")
        )
        if is_authorization_recheck and conversation.remembered_intent is None:
            conversation.remembered_intent = "general_claim_question"

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

        explicit_id = conversation.remembered_case_id
        if explicit_id and not any(
            claim["case_id"].casefold() == explicit_id.casefold() for claim in claims
        ):
            return {
                "response": "That claim reference does not match the claims available under your verified record. Please check the reference or describe the claim."
            }

        decision = self.case_handler.select_case(
            conversation=conversation,
            message=state["message"],
            claims=claims,
        )

        if decision["case_id"] is None:
            # Authorization rechecks are continuation turns. If the model
            # declines to repeat its earlier case proposal, use the already
            # remembered, unambiguous hints only when they identify one owned
            # claim; this does not broaden access or override a claim ID.
            if is_authorization_recheck:
                narrowed = claims
                if conversation.remembered_case_type:
                    narrowed = [
                        claim
                        for claim in narrowed
                        if claim.get("case_type") == conversation.remembered_case_type
                    ]
                if conversation.remembered_month:
                    narrowed = [
                        claim
                        for claim in narrowed
                        if datetime.fromisoformat(claim["created_at"]).strftime("%B")
                        == conversation.remembered_month
                    ]
                if conversation.remembered_year:
                    narrowed = [
                        claim
                        for claim in narrowed
                        if int(claim["created_at"][:4]) == conversation.remembered_year
                    ]
                if conversation.remembered_status_hint:
                    narrowed = [
                        claim
                        for claim in narrowed
                        if claim.get("status") == conversation.remembered_status_hint
                    ]
                if len(narrowed) == 1:
                    decision = {"case_id": narrowed[0]["case_id"], "question": None}
            if decision["case_id"] is None:
                conversation.pending_question = decision["question"]
                return {"response": decision["question"]}

        if explicit_id and decision["case_id"].casefold() != explicit_id.casefold():
            raise ModelResponseError(
                "Selected claim conflicts with the explicit reference."
            )

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
        """Answer follow-ups and offer a summary only when wrapping up."""

        conversation = state["conversation"]

        if state["extracted"]["conversation_action"] == "wrap_up":
            self._require_verified(conversation)
            conversation.phase = Phase.POST_PROCESS
            return self._post_process_node(state)

        # Recheck ownership before passing claim details to the handler.
        claim = self._get_selected_claim(conversation)

        answer = self.case_handler.generate_answer(
            conversation=conversation,
            message=state["message"],
            claim=claim,
            field_definitions=self.data_service.claim_schema,
            guidance=self.data_service.document_guidance,
        )

        return {"conversation": conversation, "response": answer}

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
            conversation.phase = (
                Phase.PROCESS_CASE
                if conversation.selected_case_id
                else Phase.RESOLVE_INTENT
            )
            state["extracted"]["conversation_action"] = "continue"

            return {"conversation": conversation}

        if action == "skip":
            conversation.email_consent = False
            conversation.summary_delivery_status = "skipped"

            return {
                "conversation": conversation,
                "response": (
                    "Understood. We’ll skip the email summary. "
                    "Thank you for contacting insurance support."
                ),
            }

        if action == "unclear":
            conversation.pending_field = "email_consent"
            conversation.pending_question = (
                "Would you like the email summary, or should we skip it?"
            )
            return {
                "response": (
                    "Would you like an email summary of what we discussed and the next steps, "
                    "or should we skip it? Email delivery is simulated in this demo."
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
        conversation.summary_delivery_status = "preview_generated"
        conversation.summary_preview = body

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
            raise ValueError("Representative authorization has not been approved.")

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
