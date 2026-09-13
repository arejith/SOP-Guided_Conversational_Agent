"""Script model decisions without pretending to test language understanding."""

import json
from collections import deque
from copy import deepcopy

from services.data_service import InsuranceDataService
from services.llm_service import LLMService

PII = {"name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472"}


def extraction(**updates):
    """Build one complete semantic decision from a scripted model."""

    result = {
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
        "conversation_action": "continue",
    }
    result.update(updates)
    return result


class ScriptedModel:
    """Return explicit semantic decisions; fail on unexpected extra calls."""

    required_text = staticmethod(LLMService.required_text)

    def __init__(self):
        self.extractions = deque()
        self.decisions = deque()
        self.contexts = []
        self.tasks = []

    def ask_model(self, prompt, schema=None):
        self.contexts.append(json.loads(prompt.split("Conversation data:\n", 1)[1]))
        result = self.extractions.popleft()
        if isinstance(result, Exception):
            raise result
        return json.dumps(result)

    def ask_json(self, instructions, data):
        self.tasks.append((instructions, deepcopy(data)))
        result = self.decisions.popleft()
        if isinstance(result, Exception):
            raise result
        return result


class RecordingData(InsuranceDataService):
    """Record private service calls so tests assert gate ordering."""

    def __init__(self):
        super().__init__()
        self.access = []

    def find_claims_for_party(self, party_id):
        self.access.append(("list", party_id))
        return super().find_claims_for_party(party_id)

    def find_claim_for_party(self, party_id, case_id):
        self.access.append(("get", party_id, case_id))
        return super().find_claim_for_party(party_id, case_id)
