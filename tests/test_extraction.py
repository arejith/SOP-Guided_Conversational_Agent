"""Validate extraction contracts and ensure Python never manufactures PII."""

import json

import pytest

from agent.extraction import extract_customer_information, extraction_schema
from agent.state import CallerRole, ConversationState
from agent.verification import IdentityVerifier
from services.errors import ModelResponseError
from tests.support import PII, RecordingData, extraction


@pytest.mark.parametrize(
    "message,role",
    [
        ("My SSN last four is 4472", CallerRole.UNKNOWN),
        ("My SSN last four is 4472", CallerRole.REPRESENTATIVE),
        ("Her SSN last four is 4472", CallerRole.REPRESENTATIVE),
        ("Example: 'SSN last four is 4472'", CallerRole.POLICYHOLDER),
        ("My SSN last four is not 4472", CallerRole.POLICYHOLDER),
        ("My SSN last four is 4472 or 9180", CallerRole.POLICYHOLDER),
        ("Forget my SSN last four is 4472", CallerRole.POLICYHOLDER),
        ("The unrelated DOB is 1985-03-15, policy POL-9921", CallerRole.POLICYHOLDER),
    ],
)
def test_omission_is_never_overridden_by_regex(message, role):
    state = ConversationState(caller_role=role)
    result = extract_customer_information(
        state, message, lambda _: json.dumps(extraction())
    )
    assert result["identity_fields"] == {}


def test_retraction_without_replacement_stays_retracted():
    result = extract_customer_information(
        ConversationState(),
        "Retract SSN last four is 4472",
        lambda _: json.dumps(extraction(clear_fields=["identity_fields.id_last4"])),
    )
    assert result["clear_fields"] == ["identity_fields.id_last4"]
    assert result["identity_fields"] == {}


def test_explicit_semantic_correction_replaces_old_value():
    result = extract_customer_information(
        ConversationState(),
        "It is 4472, correction to 9180",
        lambda _: json.dumps(
            extraction(
                identity_fields={"id_last4": "9180"},
                clear_fields=["identity_fields.id_last4"],
            )
        ),
    )
    assert result["identity_fields"] == {"id_last4": "9180"}
    assert result["clear_fields"] == []


@pytest.mark.parametrize("value", ["123-45-4472", "123454472"])
def test_full_ssn_cannot_be_truncated(value):
    result = extract_customer_information(
        ConversationState(),
        "My SSN is " + value,
        lambda _: json.dumps(extraction(identity_fields={"id_last4": "4472"})),
    )
    assert "id_last4" not in result["identity_fields"]
    assert "identity_fields.id_last4" in result["clear_fields"]
    assert result["clarification_question"]


@pytest.mark.parametrize(
    "value,expected",
    [
        ("March 15th, 1985", "1985-03-15"),
        ("15 March 1985", "1985-03-15"),
        ("1985-03-15", "1985-03-15"),
    ],
)
def test_natural_dates_and_pending_question_context(value, expected):
    state = ConversationState(
        caller_role=CallerRole.POLICYHOLDER,
        pending_field="dob",
        pending_question="What is your date of birth?",
    )
    prompts = []

    def ask(prompt):
        prompts.append(prompt)
        return json.dumps(extraction(identity_fields={"dob": value}))

    result = extract_customer_information(state, value, ask)
    context = json.loads(prompts[0].split("Conversation data:\n", 1)[1])
    assert context["pending_field"] == "dob"
    assert context["caller_context"]["caller_role"] == "policyholder"
    assert result["identity_fields"]["dob"] == expected


@pytest.mark.parametrize(
    "identity",
    [
        {"dob": "03/04/1985"},
        {"dob": "1985-02-30"},
        {"id_last4": "123454472"},
        {"id_last4": "447"},
        {"phone": "abc"},
        {"email": "bad-address"},
    ],
)
def test_invalid_formats_fail_closed(identity):
    with pytest.raises(ModelResponseError):
        extract_customer_information(
            ConversationState(),
            "message",
            lambda _: json.dumps(extraction(identity_fields=identity)),
        )


@pytest.mark.parametrize("output", [None, [], "bad json", {"identity_fields": {}}])
def test_malformed_output_is_recoverable(output):
    with pytest.raises(ModelResponseError):
        extract_customer_information(
            ConversationState(), "message", lambda _: json.dumps(output)
        )


def test_structured_null_identity_values_do_not_retract_memory():
    identity = {
        name: None
        for name in extraction_schema()["properties"]["identity_fields"]["required"]
    }
    result = extract_customer_information(
        ConversationState(identity_fields=PII),
        "hello",
        lambda _: json.dumps(extraction(identity_fields=identity)),
    )
    assert result["identity_fields"] == {} and result["clear_fields"] == []


def test_verifier_does_not_choose_between_duplicate_matching_records():
    data = RecordingData()
    data.policyholders.append(dict(data.policyholders[0], party_id="duplicate"))
    assert IdentityVerifier(data).find_verified_policyholder(PII) is None


def test_email_punctuation_is_significant():
    assert IdentityVerifier._normalize_value(
        "email", "a.b@example.com"
    ) != IdentityVerifier._normalize_value("email", "ab@example.com")


def test_conflicting_values_clear_old_memory_and_stop_before_verification():
    from agent.sop import InsuranceSOPGraph
    from tests.support import ScriptedModel

    state = ConversationState(
        caller_role=CallerRole.POLICYHOLDER, identity_fields=PII.copy()
    )
    model, data = ScriptedModel(), RecordingData()
    model.extractions.append(
        extraction(
            clear_fields=["identity_fields.id_last4"],
            clarification_question="Which SSN last four should I use?",
        )
    )
    graph = InsuranceSOPGraph(data, model)
    response = graph.handle_message(state, "4472 or 9180; I am not sure")
    assert "Which SSN" in response and "id_last4" not in state.identity_fields
    assert not state.is_verified and not data.access
