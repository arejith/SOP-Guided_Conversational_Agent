"""Demonstrate all four real graph phases using labelled scripted model decisions.

Run with: python -m tests.demo_workflow
This is a controller demonstration, not a live-model quality evaluation.
"""

from agent.sop import InsuranceSOPGraph
from agent.state import ConversationState, Phase
from tests.support import RecordingData, ScriptedModel, extraction


def main() -> None:
    """Print a deterministic transcript and assert each required phase."""

    model, data, state = ScriptedModel(), RecordingData(), ConversationState()
    graph = InsuranceSOPGraph(data, model)
    turns = [
        (
            "I'm the policyholder, Margaret Chen. My DOB is March 15, 1985.",
            extraction(
                caller_role="policyholder",
                identity_fields={"name": "Margaret Chen", "dob": "1985-03-15"},
            ),
            [],
            Phase.VERIFY_ID,
        ),
        (
            "My SSN last four is 4472.",
            extraction(identity_fields={"id_last4": "4472"}),
            [],
            Phase.RESOLVE_INTENT,
        ),
        (
            "Why was my January 2026 healthcare claim denied?",
            extraction(
                intent="denial_question",
                case_type="healthcare",
                month="January",
                year=2026,
            ),
            [
                {"case_id": "CL-2048", "question": None},
                {
                    "answer": "The record says the review file lacked the pathology report and treating provider office note. The recorded appeal deadline was March 18, 2026, which has passed. These fixtures do not say whether a late appeal can be accepted; human clarification is needed."
                },
            ],
            Phase.PROCESS_CASE,
        ),
        (
            "That's everything, thanks.",
            extraction(conversation_action="wrap_up"),
            [{"action": "unclear"}],
            Phase.POST_PROCESS,
        ),
        (
            "Yes, please send the email summary.",
            extraction(conversation_action="wrap_up"),
            [
                {"action": "send"},
                {
                    "summary": "We discussed the denial of CL-2048. The record lists a missing pathology report and treating provider office note. Its March 18, 2026 appeal deadline has passed. Ask a human claims representative whether any late-appeal options exist; acceptance is not established by these fixtures."
                },
            ],
            Phase.POST_PROCESS,
        ),
    ]
    print(
        "SCRIPTED MODEL DEMONSTRATION — real graph and fixtures; no API calls or email delivery.\n"
    )
    for message, info, decisions, phase in turns:
        model.extractions.append(info)
        model.decisions.extend(decisions)
        response = graph.handle_message(state, message)
        assert state.phase == phase
        print(f"Customer: {message}\nAgent [{state.phase.value}]: {response}\n")
    assert state.email_consent is True
    assert state.summary_delivery_status == "preview_generated"
    print("All four phases demonstrated. Summary delivery: preview_generated.")


if __name__ == "__main__":
    main()
