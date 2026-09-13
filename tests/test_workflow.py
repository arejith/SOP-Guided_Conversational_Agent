"""Exercise the real graph with scripted semantic decisions and service spies."""

from copy import deepcopy

import pytest

from agent.sop import InsuranceSOPGraph
from agent.state import CallerRole, ConversationState, Phase
from services.authorization_service import RepresentativeAuthorizationService
from services.errors import ModelResponseError
from tests.support import PII, RecordingData, ScriptedModel, extraction


@pytest.fixture
def setup():
    data, model, state = RecordingData(), ScriptedModel(), ConversationState()
    graph = InsuranceSOPGraph(data, model)
    return graph, state, data, model


def turn(setup, info, *decisions, message="Customer message"):
    graph, state, _, model = setup
    model.extractions.append(info)
    model.decisions.extend(decisions)
    return graph.handle_message(state, message)


def verified(setup):
    return turn(
        setup,
        extraction(
            caller_role="policyholder",
            identity_fields=PII,
            intent="denial_question",
            case_type="healthcare",
            month="January",
            reported_status="denied",
        ),
        {"case_id": "CL-2048", "question": None},
        {"answer": "The record lists a missing pathology report and office note."},
    )


def representative(setup):
    return turn(
        setup,
        extraction(
            caller_role="representative",
            representative_name="David Chen",
            representative_relationship="son",
            identity_fields=PII,
            intent="denial_question",
            case_type="healthcare",
            month="January",
        ),
    )


def test_partial_identity_remembers_intent_and_pending_question(setup):
    _, state, data, model = setup
    response = turn(
        setup,
        extraction(
            caller_role="policyholder",
            identity_fields={
                "name": "Margaret Chen",
                "dob": "March 15th, 1985",
                "policy_number": "POL-9921",
            },
            intent="denial_question",
            month="January",
        ),
    )
    assert not state.is_verified and not data.access
    assert state.pending_field == "phone"
    assert "phone" in response and "full name" not in response
    turn(
        setup,
        extraction(identity_fields={"id_last4": "4472"}),
        {"case_id": "CL-2048", "question": None},
        {"answer": "Recorded denial reason."},
        message="SSN last four is 4472",
    )
    assert model.contexts[-1]["pending_field"] == "phone"
    assert state.selected_case_id == "CL-2048"
    assert state.remembered_intent == "denial_question"
    assert state.phase == Phase.PROCESS_CASE


def test_unknown_role_does_not_authorize_three_pii(setup):
    turn(setup, extraction(identity_fields=PII))
    _, state, data, _ = setup
    assert not state.is_verified and not data.access
    assert state.pending_field == "caller_role"


def test_ambiguous_caller_gets_focused_question_and_can_recover(setup):
    response = turn(
        setup,
        extraction(
            identity_fields=PII,
            identity_ambiguous=True,
            clarification_question="Who is speaking, and who is the policyholder?",
        ),
    )
    _, state, data, _ = setup
    assert response.startswith("Who is speaking")
    assert state.identity_fields == {} and not data.access
    representative(setup)
    assert state.caller_role == CallerRole.REPRESENTATIVE
    assert state.authorization_request_id
    assert not data.access


@pytest.mark.parametrize(
    "topic", ["verification_reason", "identity_options", "document_preparation"]
)
def test_general_guidance_has_no_private_access(setup, topic):
    response = turn(setup, extraction(general_question=topic))
    _, state, data, model = setup
    assert response and state.phase == Phase.VERIFY_ID and not data.access
    assert not model.tasks
    assert "CL-" not in response and "{documents}" not in response


@pytest.mark.parametrize(
    "emotion", ["frustration", "anger", "anxiety", "confusion", "refusal"]
)
def test_empathy_keeps_gate_closed(setup, emotion):
    response = turn(setup, extraction(caller_role="policyholder", emotion=emotion))
    assert response and not setup[1].is_verified and not setup[2].access
    assert any(
        word in response.lower() for word in ["understand", "hear", "happy", "choose"]
    )


def test_unrelated_attempts_offer_assistance_without_trapping_session(setup):
    for _ in range(3):
        response = turn(setup, extraction(scope="out_of_scope"))
    assert "human assistance" in response
    assert not setup[2].access
    verified(setup)
    assert setup[1].selected_case_id == "CL-2048"


def test_human_request_does_not_claim_transfer_or_lock_session(setup):
    response = turn(setup, extraction(human_requested=True))
    assert "cannot connect" in response
    verified(setup)
    assert setup[1].is_verified


def test_authorization_pending_approval_and_remembered_request(setup):
    response = representative(setup)
    graph, state, data, model = setup
    assert "pending" in response and not data.access
    turn(
        setup,
        extraction(),
        {"case_id": "CL-2048", "question": None},
        {"answer": "The recorded denial concerns missing documents."},
        message="Check authorization again",
    )
    assert state.phase == Phase.PROCESS_CASE
    assert (
        graph.authorization_service.requests[state.authorization_request_id].status
        == "approved"
    )
    context = model.tasks[-1][1]
    assert context["access_context"]["claim_access_authorized"] is True
    assert context["intent"] == "denial_question"


@pytest.mark.parametrize(
    "scenario,expected", [("denied", "denied"), ("timeout", "timed out")]
)
def test_terminal_authorization_never_accesses_claims(setup, scenario, expected):
    graph, state, data, _ = setup
    graph.authorization_service = RepresentativeAuthorizationService(data, scenario)
    representative(setup)
    for _ in range(max(1, len(graph.authorization_service.status_sequence) - 1)):
        response = turn(setup, extraction())
    assert expected in response and not data.access
    assert state.phase == Phase.VERIFY_ID


def test_refusal_does_not_poll_authorization(setup):
    representative(setup)
    graph, state, data, _ = setup
    request = graph.authorization_service.requests[state.authorization_request_id]
    polls = request.polls
    turn(setup, extraction(emotion="refusal"))
    assert request.polls == polls and not data.access


def test_representative_to_policyholder_drops_all_access(setup):
    representative(setup)
    graph, state, data, _ = setup
    old_request = state.authorization_request_id
    turn(setup, extraction(caller_role="policyholder"), message="I am the policyholder")
    assert state.identity_fields == {} and not state.is_verified
    assert old_request not in graph.authorization_service.requests
    assert not data.access and state.remembered_intent is None


def test_new_policyholder_before_verification_cannot_reuse_pii(setup):
    turn(
        setup,
        extraction(
            caller_role="policyholder",
            identity_fields={"name": "Margaret Chen", "dob": "1985-03-15"},
        ),
    )
    turn(
        setup,
        extraction(
            caller_role="policyholder",
            identity_context_changed=True,
            identity_fields={"name": "Ava Lopez"},
        ),
    )
    assert setup[1].identity_fields == {"name": "ava lopez"}
    assert not setup[1].is_verified


def test_formatting_change_keeps_access_and_claim_context(setup):
    verified(setup)
    turn(
        setup,
        extraction(
            identity_fields={"name": "  MARGARET   CHEN ", "dob": "March 15, 1985"}
        ),
        {"answer": "A follow-up answer."},
    )
    assert setup[1].is_verified and setup[1].selected_case_id == "CL-2048"
    assert len(setup[1].messages) == 4


def test_retraction_revokes_access_but_preserves_other_same_person_fields(setup):
    verified(setup)
    setup[2].access.clear()
    turn(setup, extraction(clear_fields=["identity_fields.id_last4"]))
    state = setup[1]
    assert not state.is_verified and not setup[2].access
    assert state.identity_fields == {"name": "margaret chen", "dob": "1985-03-15"}
    assert state.remembered_intent == "denial_question"
    assert state.selected_case_id is None
    assert all("pathology" not in msg["content"] for msg in state.messages)


def test_correction_requires_reverification_and_can_recover(setup):
    verified(setup)
    setup[2].access.clear()
    turn(setup, extraction(identity_fields={"dob": "1985-03-16"}))
    assert not setup[1].is_verified and not setup[2].access
    turn(
        setup,
        extraction(identity_fields={"dob": "1985-03-15"}),
        {"case_id": "CL-2048", "question": None},
        {"answer": "Recovered."},
    )
    assert setup[1].is_verified


def test_explicit_cross_party_reference_never_silently_substitutes(setup):
    response = turn(
        setup,
        extraction(
            caller_role="policyholder",
            identity_fields=PII,
            intent="status_inquiry",
            case_id="CL-3001",
        ),
    )
    assert "check the reference" in response
    assert not setup[3].tasks
    assert setup[1].selected_case_id is None
    assert all(call[1] == "P9" for call in setup[2].access)


def test_model_cannot_select_another_policyholders_claim(setup):
    with pytest.raises(ValueError, match="not authorized"):
        turn(
            setup,
            extraction(
                caller_role="policyholder", identity_fields=PII, intent="status_inquiry"
            ),
            {"case_id": "CL-3001", "question": None},
        )
    assert not setup[1].is_verified
    assert not any("claim" in context for _, context in setup[3].tasks)


def test_ambiguous_claim_question_then_select(setup):
    response = turn(
        setup,
        extraction(
            caller_role="policyholder",
            identity_fields=PII,
            intent="status_inquiry",
            case_type="healthcare",
            month="January",
        ),
        {"case_id": None, "question": "January 2025 or January 2026?"},
    )
    assert response == "January 2025 or January 2026?"
    assert setup[1].selected_case_id is None
    turn(
        setup,
        extraction(year=2026),
        {"case_id": "CL-2048", "question": None},
        {"answer": "Recorded status."},
    )
    assert setup[1].selected_case_id == "CL-2048"


def test_followup_keeps_selected_claim_and_no_summary_offer(setup):
    response = verified(setup)
    assert "email" not in response and setup[1].phase == Phase.PROCESS_CASE
    turn(
        setup,
        extraction(intent="document_submission"),
        {"answer": "I cannot confirm receipt from this record."},
    )
    assert setup[1].phase == Phase.PROCESS_CASE
    assert "claim" in setup[3].tasks[-1][1]


def test_new_claim_hint_forces_selection(setup):
    verified(setup)
    turn(
        setup,
        extraction(case_id="CL-2102", intent="status_inquiry"),
        {"case_id": "CL-2102", "question": None},
        {"answer": "The auto claim is open."},
    )
    assert setup[1].selected_case_id == "CL-2102"


def test_wrap_up_uncertain_and_continued_questions(setup):
    verified(setup)
    response = turn(
        setup,
        extraction(conversation_action="wrap_up"),
        {"action": "unclear"},
        message="That's all, thanks",
    )
    assert "email summary" in response and setup[1].phase == Phase.POST_PROCESS
    turn(setup, extraction(), {"action": "unclear"}, message="Maybe")
    assert setup[1].email_consent is None
    turn(
        setup,
        extraction(intent="next_steps"),
        {"action": "followup"},
        {"answer": "Next steps."},
    )
    assert setup[1].phase == Phase.PROCESS_CASE and setup[1].email_consent is None


@pytest.mark.parametrize("action", ["send", "skip"])
def test_explicit_summary_choice_respected_and_delivery_separate(setup, action):
    verified(setup)
    decisions = [{"action": action}]
    if action == "send":
        decisions.append(
            {"summary": "We discussed the recorded denial and missing documents."}
        )
    response = turn(setup, extraction(conversation_action="wrap_up"), *decisions)
    assert setup[1].email_consent is (action == "send")
    assert setup[1].summary_delivery_status == (
        "preview_generated" if action == "send" else "skipped"
    )
    if action == "send":
        assert "No email was actually sent" in response


def test_supplied_email_is_not_consent(setup):
    verified(setup)
    turn(setup, extraction(conversation_action="wrap_up"), {"action": "unclear"})
    turn(setup, extraction(), {"action": "unclear"}, message="another@example.com")
    assert setup[1].email_consent is None
    assert setup[1].summary_delivery_status == "not_requested"


def test_failed_model_rolls_back_conversation_and_authorization_poll(setup):
    representative(setup)
    graph, state, _, _ = setup
    before, requests = deepcopy(state), deepcopy(graph.authorization_service.requests)
    response = turn(setup, extraction(), ModelResponseError("bad response"))
    assert "couldn’t complete" in response
    assert state == before and graph.authorization_service.requests == requests


def test_failed_new_identity_cannot_restore_previous_access(setup):
    verified(setup)
    response = turn(
        setup,
        extraction(
            identity_context_changed=True,
            caller_role="policyholder",
            identity_fields=PII,
            intent="status_inquiry",
        ),
        ModelResponseError("bad response"),
    )
    assert "verify the current caller" in response
    assert not setup[1].is_verified and not setup[1].messages


def test_programming_error_is_not_swallowed(setup):
    with pytest.raises(TypeError):
        turn(setup, TypeError("programming bug"))


def test_general_question_preserves_pending_identity_question(setup):
    turn(
        setup,
        extraction(
            caller_role="policyholder", identity_fields={"name": "Margaret Chen"}
        ),
    )
    question = setup[1].pending_question
    assert setup[1].pending_field == "dob"
    turn(setup, extraction(general_question="verification_reason"))
    assert setup[1].pending_field == "dob" and setup[1].pending_question == question
    turn(
        setup,
        extraction(identity_fields={"dob": "1985-03-15"}),
        message="March 15, 1985",
    )
    assert setup[3].contexts[-1]["pending_field"] == "dob"
    assert setup[1].identity_fields["dob"] == "1985-03-15"


def test_diagnostics_do_not_log_error_message_or_pii(setup, caplog):
    turn(setup, ModelResponseError("secret-key Margaret Chen 4472"))
    assert "ModelResponseError" in caplog.text
    assert "secret-key" not in caplog.text and "4472" not in caplog.text
