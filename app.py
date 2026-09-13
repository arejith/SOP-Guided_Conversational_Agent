"""Session-isolated Streamlit chat for the fixture-based insurance demo."""

import logging
from collections.abc import MutableMapping
from typing import Any

import streamlit as st

from agent.sop import InsuranceSOPGraph
from agent.state import CallerRole, ConversationState, Phase
from services.data_service import InsuranceDataService
from services.llm_service import LLMService

logger = logging.getLogger(__name__)


def reset_session(session: MutableMapping[str, Any]) -> None:
    """Close this session's client and discard its mutable conversation state."""

    client = session.pop("llm", None)
    if client is not None:
        client.close()
    for name in ("agent", "active_configuration", "recovery_message"):
        session.pop(name, None)
    session["conversation"] = ConversationState()


def configure_session(
    session: MutableMapping[str, Any], api_key: str, model: str
) -> None:
    """Recreate services when this session's credentials or model change."""

    configuration = (api_key.strip(), model.strip())
    if session.get("active_configuration") == configuration:
        return
    reset_session(session)
    client = LLMService(api_key=configuration[0], model=configuration[1])
    try:
        agent = InsuranceSOPGraph(InsuranceDataService(), client)
    except Exception:
        client.close()
        raise
    session["llm"] = client
    session["agent"] = agent
    session["active_configuration"] = configuration


def access_label(
    agent: InsuranceSOPGraph | None, conversation: ConversationState
) -> str:
    """Distinguish PII verification from representative authorization."""

    if not conversation.is_verified:
        return "Identity verification needed"
    if conversation.caller_role == CallerRole.POLICYHOLDER:
        return "Identity verified · claim access allowed"
    if agent and conversation.authorization_request_id:
        request = agent.authorization_service.requests.get(
            conversation.authorization_request_id
        )
        if request:
            return f"Policyholder identity verified · representative authorization {request.status} (simulated)"
    return "Policyholder identity verified · representative authorization needed"


def main() -> None:
    """Render configuration, workflow status, and the current session's chat."""

    st.set_page_config(page_title="Insurance claims support", page_icon="💬")
    st.title("Insurance claims support")
    st.caption("A guided claims conversation using fictional records.")
    st.info(
        "Demo: email is a preview, representative approval is simulated, and live human transfer is not connected. Use fictional identity details only."
    )
    if "conversation" not in st.session_state:
        st.session_state.conversation = ConversationState()
    defaults = LLMService.configuration()
    with st.sidebar:
        st.subheader("Connection")
        key = st.text_input(
            "OpenAI API key",
            type="password",
            key="api_key_input",
            help="Kept in this browser session's server memory. Leave blank to use server configuration.",
        )
        model = st.text_input("Model", value=defaults["model"], key="model_input")
        effective_key = key.strip() or defaults["api_key"]
        current = (effective_key.strip(), model.strip())
        if st.session_state.get("active_configuration") not in (None, current):
            reset_session(st.session_state)
        if st.button(
            "Connect", disabled=not effective_key.strip() or not model.strip()
        ):
            try:
                configure_session(st.session_state, effective_key, model)
            except Exception as error:
                logger.error("Client setup failed: error_type=%s", type(error).__name__)
                st.error(
                    "Could not initialize the connection. Check your configuration and try again."
                )
        if st.button("New conversation"):
            reset_session(st.session_state)
            st.rerun()
        conversation = st.session_state.conversation
        st.subheader("Workflow")
        phase_order = list(Phase)
        phase_position = phase_order.index(conversation.phase)
        st.progress(
            (phase_position + 1) / len(phase_order),
            text=f"Phase {phase_position + 1} of {len(phase_order)}",
        )
        st.caption(" → ".join(phase.value for phase in phase_order))
        st.write(conversation.phase.value)
        if "agent" in st.session_state:
            st.success("Connected · API token is kept hidden in this session")
        elif effective_key.strip():
            st.info("Token configured · select Connect")
        else:
            st.warning("No API token configured")
        st.caption(access_label(st.session_state.get("agent"), conversation))
        st.caption(
            "Consent: "
            + {None: "undecided", True: "agreed", False: "declined"}[
                conversation.email_consent
            ]
        )
        st.caption("Summary: " + conversation.summary_delivery_status.replace("_", " "))

    conversation = st.session_state.conversation
    if not conversation.messages:
        with st.chat_message("assistant"):
            st.write(
                "How can I help with your insurance claim? Please tell me whether you are the policyholder or calling for someone else. To discuss a claim, we’ll need three permitted identity details."
            )
    for message in conversation.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    if st.session_state.get("recovery_message"):
        st.warning(st.session_state.recovery_message)
    connected = "agent" in st.session_state
    if not connected:
        st.caption(
            "Enter an API key in the sidebar or configure the server, then select Connect."
        )
    message = st.chat_input(
        "Your message", disabled=not connected or conversation.email_consent is not None
    )
    if message:
        try:
            with st.spinner("Working on your request…"):
                response = st.session_state.agent.handle_message(conversation, message)
            # Recoverable failures intentionally do not enter conversation memory.
            committed = bool(
                conversation.messages
                and conversation.messages[-1]["content"] == response
            )
            st.session_state.recovery_message = None if committed else response
        except Exception as error:
            logger.error(
                "Workflow failed: phase=%s error_type=%s",
                conversation.phase.value,
                type(error).__name__,
            )
            st.session_state.recovery_message = (
                "Something went wrong. Please try again or start a new conversation."
            )
        st.rerun()


if __name__ == "__main__":
    main()
