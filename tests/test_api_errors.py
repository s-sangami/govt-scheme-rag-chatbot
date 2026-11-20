import httpx
import groq
import pytest
from streamlit.testing.v1 import AppTest

TOO_LARGE = "Your question needs too much context, try a shorter question."
RATE_LIMIT = "Daily limit reached, try again later."


def _error(cls, status):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return cls("Request too large ... TPM limit 8000 (raw api detail)", response=response, body=None)


class RaisingChain:
    """Fake chain: never calls the real API, just raises the given error."""

    def __init__(self, exc):
        self.exc = exc

    def invoke(self, inputs):
        raise self.exc


def _run(exc):
    at = AppTest.from_file("app.py", default_timeout=120)
    at.session_state["pdf_processed"] = True
    at.session_state["chat_history"] = []
    at.session_state["qa_chain"] = RaisingChain(exc)
    at.session_state["chunk_count"] = 1
    at.session_state["char_count"] = 1
    at.run()
    at.chat_input[0].set_value("How much money do farmers get per year?").run()
    return at


@pytest.mark.parametrize("exc,expected", [
    (_error(groq.APIStatusError, 413), TOO_LARGE),
    (_error(groq.RateLimitError, 429), RATE_LIMIT),
])
def test_api_error_shows_friendly_message_without_traceback(exc, expected):
    at = _run(exc)
    assert not at.exception  # no raw traceback rendered
    assert [e.value for e in at.error] == [expected]
    last = at.session_state["chat_history"][-1]
    assert last["role"] == "assistant"
    assert last["content"] == expected
    assert "TPM limit" not in str(at.get("error")[0].value)
