"""Session lifecycle, credential isolation, and model failure boundaries."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from streamlit.proto.TextInput_pb2 import TextInput
from streamlit.testing.v1 import AppTest

from app import configure_session, reset_session
from services.errors import ModelResponseError
from services.llm_service import LLMService


def test_separate_sessions_own_clients_graphs_and_state(monkeypatch):
    first, second = {}, {}
    clients = [MagicMock(), MagicMock(), MagicMock()]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("app.LLMService", side_effect=clients) as factory:
        configure_session(first, "session-one", "model-one")
        configure_session(second, "session-two", "model-two")
        assert first["agent"] is not second["agent"]
        assert first["conversation"] is not second["conversation"]
        first["conversation"].identity_fields["name"] = "One"
        assert second["conversation"].identity_fields == {}
        configure_session(first, "session-one", "model-one")
        assert factory.call_count == 2
        configure_session(first, "changed-key", "changed-model")
        clients[0].close.assert_called_once()
        assert first["conversation"].identity_fields == {}
        assert second["llm"] is clients[1]
        reset_session(first)
        clients[2].close.assert_called_once()
        assert "agent" not in first
    import os

    assert "OPENAI_API_KEY" not in os.environ


def test_explicit_configuration_overrides_defaults_without_environment_mutation(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "server-key")
    monkeypatch.setenv("OPENAI_MODEL", "server-model")
    with patch("services.llm_service.OpenAI") as client:
        service = LLMService(api_key="session-key", model="session-model")
        assert client.call_args.kwargs["api_key"] == "session-key"
        assert service.model == "session-model"
    import os

    assert os.environ["OPENAI_API_KEY"] == "server-key"
    assert os.environ["OPENAI_MODEL"] == "server-model"


def test_dotenv_defaults_do_not_mutate_environment(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with patch(
        "services.llm_service.dotenv_values",
        return_value={"OPENAI_API_KEY": "local-key", "OPENAI_MODEL": "local-model"},
    ):
        assert LLMService.configuration() == {
            "api_key": "local-key",
            "model": "local-model",
        }
    import os

    assert "OPENAI_API_KEY" not in os.environ


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(status="incomplete", output_text="{}"),
        SimpleNamespace(status="completed", output_text=""),
    ],
)
def test_incomplete_or_empty_api_output_is_recoverable(response):
    with patch("services.llm_service.OpenAI") as client:
        client.return_value.responses.create.return_value = response
        service = LLMService(api_key="test-key")
        with pytest.raises(ModelResponseError):
            service.ask_model("Instructions")


def test_extraction_uses_strict_schema_and_disables_storage():
    from agent.extraction import extraction_schema

    with patch("services.llm_service.OpenAI") as client:
        client.return_value.responses.create.return_value = SimpleNamespace(
            status="completed", output_text="{}"
        )
        service = LLMService(api_key="test-key")
        service.ask_model("Instructions", schema=extraction_schema())
        arguments = client.return_value.responses.create.call_args.kwargs
        assert arguments["store"] is False
        assert arguments["text"]["format"]["strict"] is True


def test_streamlit_starts_without_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("services.llm_service.dotenv_values", return_value={}):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    assert not app.exception
    assert app.chat_input[0].disabled
    assert app.text_input[0].proto.type == TextInput.PASSWORD


def test_streamlit_key_change_closes_client_and_disables_chat(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("services.llm_service.dotenv_values", return_value={}):
        # AppTest executes app.py under its own module name; patch the API client.
        with patch("services.llm_service.OpenAI") as client:
            app = AppTest.from_file(
                Path(__file__).resolve().parents[1] / "app.py"
            ).run()
            app.text_input[0].set_value("session-key").run()
            app.button[0].click().run()
            assert not app.exception and not app.chat_input[0].disabled
            app.text_input[0].set_value("replacement-key").run()
            assert app.chat_input[0].disabled
            client.return_value.close.assert_called_once()


def test_streamlit_chat_runs_all_four_phases(monkeypatch):
    from agent.state import Phase
    from tests.support import ScriptedModel, extraction

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    model = ScriptedModel()
    model.extractions.extend(
        [
            extraction(
                caller_role="policyholder",
                identity_fields={"name": "Margaret Chen", "dob": "1985-03-15"},
            ),
            extraction(identity_fields={"id_last4": "4472"}),
            extraction(intent="denial_question", case_id="CL-2048"),
            extraction(conversation_action="wrap_up"),
            extraction(conversation_action="wrap_up"),
        ]
    )
    model.decisions.extend(
        [
            {"case_id": "CL-2048", "question": None},
            {"answer": "The record lists missing documents."},
            {"action": "unclear"},
            {"action": "skip"},
        ]
    )
    with (
        patch("services.llm_service.dotenv_values", return_value={}),
        patch("services.llm_service.OpenAI"),
        patch.object(LLMService, "ask_model", side_effect=model.ask_model),
        patch.object(LLMService, "ask_json", side_effect=model.ask_json),
    ):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
        app.text_input[0].set_value("test-only-key").run()
        app.button[0].click().run()
        for message, phase in [
            ("Margaret, DOB March 15, 1985", Phase.VERIFY_ID),
            ("SSN last four 4472", Phase.RESOLVE_INTENT),
            ("Why was CL-2048 denied?", Phase.PROCESS_CASE),
            ("That's all", Phase.POST_PROCESS),
            ("Skip", Phase.POST_PROCESS),
        ]:
            app.chat_input[0].set_value(message).run()
            assert not app.exception
            assert app.session_state["conversation"].phase == phase
        assert app.session_state["conversation"].email_consent is False
        assert app.chat_input[0].disabled
