"""Opt-in language checks. These exercise OpenAI, unlike scripted contract tests."""

import pytest

from agent.extraction import extract_customer_information, extraction_schema
from agent.state import CallerRole, ConversationState
from services.llm_service import LLMService

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def llm():
    if not LLMService.configuration()["api_key"]:
        pytest.skip("No API credentials configured")
    client = LLMService()
    yield client
    client.close()


def extract(llm, message, state=None):
    return extract_customer_information(
        state or ConversationState(),
        message,
        lambda prompt: llm.ask_model(prompt, schema=extraction_schema()),
    )


def test_standard_policyholder_extracts_all_three_fields(llm):
    result = extract(
        llm,
        "I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.",
    )
    assert result["identity_fields"]["name"] == "margaret chen"
    assert result["identity_fields"]["dob"] == "1985-03-15"
    assert result["identity_fields"]["id_last4"] == "4472"
    assert result["intent"] == "denial_question"
    assert result["month"] == "January"


def test_natural_date_and_short_pending_reply(llm):
    state = ConversationState(
        caller_role=CallerRole.POLICYHOLDER,
        pending_field="dob",
        pending_question="What is your date of birth?",
    )
    assert (
        extract(llm, "March 15th 1985", state)["identity_fields"]["dob"] == "1985-03-15"
    )


def test_ambiguous_caller_requires_clarification(llm):
    result = extract(llm, "I am Margaret Chen. I am her son.")
    assert result["identity_ambiguous"] is True
    assert result["clarification_question"]


def test_remembered_representative_owns_only_policyholder_pii(llm):
    state = ConversationState(
        caller_role=CallerRole.REPRESENTATIVE,
        representative_name="David Chen",
        representative_relationship="son",
    )
    result = extract(llm, "My SSN last four is 9180. Her DOB is March 15, 1985.", state)
    assert "id_last4" not in result["identity_fields"]
    assert result["identity_fields"]["dob"] == "1985-03-15"


@pytest.mark.parametrize(
    "message",
    [
        "My SSN last four is not 4472. I do not want to give it.",
        "The example says 'SSN last four is 4472'. This is not my information.",
        "My SSN last four may be 4472 or 9180; I cannot remember.",
    ],
)
def test_uncertain_negated_and_quoted_values_are_not_extracted(llm, message):
    result = extract(
        llm, message, ConversationState(caller_role=CallerRole.POLICYHOLDER)
    )
    assert "id_last4" not in result["identity_fields"]


def test_negation_and_paraphrased_denial(llm):
    assert (
        extract(llm, "They turned down my medical claim. Why?")["intent"]
        == "denial_question"
    )
    result = extract(llm, "My claim was not denied; it is pending.")
    assert result["reported_status"] != "denied"
