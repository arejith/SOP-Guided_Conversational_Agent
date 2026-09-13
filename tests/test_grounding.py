"""Check factual context boundaries; live tests separately assess generated prose."""

from agent.case_handler import CaseHandler, applicable_guidance
from agent.post_process import PostProcessHandler
from agent.prompts import CLAIM_ANSWER_PROMPT, EMAIL_SUMMARY_PROMPT
from agent.state import ConversationState
from tests.support import RecordingData, ScriptedModel


def test_guidance_does_not_impose_original_report_or_new_deadline():
    data = RecordingData()
    claim = data.find_claim_for_party("P9", "CL-2048")
    guidance = applicable_guidance(claim, data.document_guidance)
    assert "original pathology report" not in guidance["document_guidance"]
    assert "treating provider office note" in guidance["document_guidance"]
    assert all(
        item["topic"] != "submission_timing"
        for item in guidance["claim_followup_guidance"]
    )
    assert "auto" not in guidance["case_type_guidance"]


def test_claim_without_missing_documents_does_not_get_missing_document_templates():
    data = RecordingData()
    claim = data.find_claim_for_party("P9", "CL-1899")
    guidance = applicable_guidance(claim, data.document_guidance)
    assert "claim_followup_guidance" not in guidance
    assert "document_alternative_guidance" not in guidance
    assert guidance["document_guidance"] == {}


def test_answer_sources_use_record_status_and_summary_uses_same_guidance():
    data, model = RecordingData(), ScriptedModel()
    state = ConversationState(remembered_status_hint="approved")
    claim = data.find_claim_for_party("P9", "CL-2048")
    model.decisions.extend(
        [
            {"answer": "Recorded status is denied."},
            {"summary": "Recorded status is denied."},
        ]
    )
    CaseHandler(model).generate_answer(
        state, "What happened?", claim, data.claim_schema, data.document_guidance
    )
    PostProcessHandler(model).generate_summary(state, claim, data.document_guidance)
    answer_context = model.tasks[0][1]
    assert answer_context["claim"]["status"] == "denied"
    assert answer_context["access_context"]["claim_access_authorized"] is True
    assert "today" in answer_context
    assert model.tasks[1][1]["guidance"] == answer_context["guidance"]
    assert "late appeals" in CLAIM_ANSWER_PROMPT
    assert "late appeals" in EMAIL_SUMMARY_PROMPT
