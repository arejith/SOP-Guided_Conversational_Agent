"""Traceable acceptance tests for the four-phase SOP harness.

These tests use scripted semantic model decisions so failures identify the
controller, gates, memory, or grounding contract rather than API variability.
Live model interpretation is covered separately by ``test_live_semantics.py``.
"""

import pytest

from agent.case_handler import applicable_guidance
from agent.state import Phase
from services.authorization_service import RepresentativeAuthorizationService
from tests.support import PII, RecordingData, ScriptedModel, extraction


@pytest.fixture
def harness():
    data, model = RecordingData(), ScriptedModel()
    from agent.sop import InsuranceSOPGraph

    return (
        InsuranceSOPGraph(data, model),
        data,
        model,
        __import__("agent.state", fromlist=["ConversationState"]).ConversationState(),
    )


def send(harness, info, *, decisions=(), message="customer message"):
    graph, _, model, state = harness
    model.extractions.append(info)
    model.decisions.extend(decisions)
    return graph.handle_message(state, message)


def verify_and_select(harness):
    return send(
        harness,
        extraction(
            caller_role="policyholder",
            identity_fields=PII,
            intent="denial_question",
            case_type="healthcare",
            month="January",
            reported_status="denied",
        ),
        decisions=(
            {"case_id": "CL-2048", "question": None},
            {"answer": "The record lists a missing pathology report and office note."},
        ),
    )


def test_standard_acceptance_follows_all_four_phases(harness):
    graph, data, model, state = harness
    response = send(
        harness,
        extraction(
            caller_role="policyholder",
            identity_fields={"name": "Margaret Chen", "dob": "1985-03-15"},
            intent="denial_question",
            case_type="healthcare",
            month="January",
            reported_status="denied",
        ),
    )
    assert state.phase == Phase.VERIFY_ID
    assert "pathology" not in response.lower()
    assert "denied" not in response.lower()
    assert not data.access

    response = send(
        harness,
        extraction(identity_fields={"id_last4": "4472"}),
        decisions=(
            {"case_id": "CL-2048", "question": None},
            {"answer": "The claim record lists missing documents."},
        ),
        message="My SSN last four is 4472.",
    )
    assert state.is_verified and state.verified_party_id == "P9"
    assert state.selected_case_id == "CL-2048"
    assert state.phase == Phase.PROCESS_CASE
    assert "missing documents" in response
    assert model.tasks[-1][1]["access_context"]["claim_access_authorized"] is True

    send(
        harness,
        extraction(conversation_action="wrap_up"),
        decisions=({"action": "skip"},),
        message="That is all, skip the summary.",
    )
    assert state.phase == Phase.POST_PROCESS
    assert state.email_consent is False


def test_policy_number_never_counts_as_one_of_three_fields(harness):
    send(
        harness,
        extraction(
            caller_role="policyholder",
            identity_fields={"name": "Margaret Chen", "policy_number": "POL-9921"},
            intent="denial_question",
        ),
    )
    _, data, _, state = harness
    assert not state.is_verified and state.phase == Phase.VERIFY_ID
    assert not data.access


def test_private_claim_data_is_never_given_before_identity_gate(harness):
    response = send(
        harness,
        extraction(intent="denial_question", case_type="healthcare", month="January"),
        message="Just tell me why my claim was denied.",
    )
    (
        _,
        data,
        state,
    ) = harness[:1] + (harness[1], harness[3])
    assert not data.access
    assert not state.is_verified
    assert "pathology" not in response.lower()


def test_emotional_recovery_acknowledges_frustration_without_bypass(harness):
    response = send(
        harness,
        extraction(
            caller_role="policyholder", emotion="frustration", intent="denial_question"
        ),
        message="I already told you who I am. This is ridiculous. Tell me why it was denied.",
    )
    _, data, _, state = harness
    assert any(
        word in response.lower() for word in ("frustrat", "understand", "protect")
    )
    assert not state.is_verified and not data.access


def test_out_of_scope_escalation_happens_after_repeated_attempts(harness):
    for _ in range(3):
        response = send(
            harness, extraction(scope="out_of_scope"), message="What is RL?"
        )
    assert "human assistance" in response.lower()
    assert not harness[3].is_verified and not harness[1].access


def test_claim_hint_memory_survives_verification_boundary(harness):
    send(
        harness,
        extraction(
            caller_role="policyholder",
            identity_fields={"name": "Margaret Chen", "dob": "1985-03-15"},
            intent="denial_question",
            case_type="healthcare",
            month="January",
        ),
    )
    state = harness[3]
    assert state.remembered_intent == "denial_question"
    send(
        harness,
        extraction(identity_fields={"id_last4": "4472"}),
        decisions=(
            {"case_id": "CL-2048", "question": None},
            {"answer": "Grounded answer."},
        ),
    )
    assert state.selected_case_id == "CL-2048"
    assert state.remembered_case_type == "healthcare"
    assert state.remembered_month == "January"


def test_follow_up_stays_in_process_case_until_wrap_up(harness):
    verify_and_select(harness)
    graph, data, model, state = harness
    response = send(
        harness,
        extraction(intent="document_submission"),
        decisions=({"answer": "I cannot confirm receipt from this record."},),
        message="Did you receive the documents?",
    )
    assert state.phase == Phase.PROCESS_CASE
    assert state.email_consent is None
    assert "confirm receipt" in response
    assert model.tasks[-1][1]["claim"]["case_id"] == "CL-2048"


def test_grounding_filters_unrelated_documents_and_deadline_rules(harness):
    data = harness[1]
    denied = data.find_claim_for_party("P9", "CL-2048")
    guidance = applicable_guidance(denied, data.document_guidance)
    assert "treating provider office note" in guidance["document_guidance"]
    assert "original pathology report" not in guidance["document_guidance"]
    assert all(
        item["topic"] != "submission_timing"
        for item in guidance["claim_followup_guidance"]
    )


@pytest.mark.parametrize(
    "scenario,expected",
    [("default", "approved"), ("denied", "denied"), ("timeout", "timeout")],
)
def test_representative_authorization_scenarios_are_bounded(
    harness, scenario, expected
):
    graph, data, _, state = harness
    graph.authorization_service = RepresentativeAuthorizationService(data, scenario)
    result = graph.authorization_service.begin_request("P9", "David Chen", "son")
    statuses = [
        graph.authorization_service.check_status(result, "P9") for _ in range(6)
    ]
    assert statuses[-1] == expected or expected in statuses
    if expected != "approved":
        assert not graph.authorization_service.is_approved(
            result, "P9", "David Chen", "son"
        )


def test_cross_policyholder_claim_is_rejected_by_controller(harness):
    graph, _, model, state = harness
    model.extractions.append(
        extraction(
            caller_role="policyholder", identity_fields=PII, intent="status_inquiry"
        )
    )
    model.decisions.append({"case_id": "CL-3001", "question": None})
    with pytest.raises(ValueError, match="not authorized"):
        graph.handle_message(state, "Show me CL-3001.")
    assert not state.messages and not state.is_verified
