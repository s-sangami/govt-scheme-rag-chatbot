from types import SimpleNamespace
from streamlit.testing.v1 import AppTest

ANSWER = "Farmers receive Rs. 6,000 per year."


class FakeChain:
    def __init__(self, answer):
        self.answer = answer

    def invoke(self, inputs):
        docs = [SimpleNamespace(metadata={"page": 3}),
                SimpleNamespace(metadata={"page": 2})]
        return {"result": self.answer, "source_documents": docs}


def _run(answer):
    at = AppTest.from_file("app.py", default_timeout=120)
    at.session_state["pdf_processed"] = True
    at.session_state["chat_history"] = []
    at.session_state["qa_chain"] = FakeChain(answer)
    at.session_state["chunk_count"] = 1
    at.session_state["char_count"] = 1
    at.run()
    at.chat_input[0].set_value("How much money do farmers get per year?").run()
    return at


def test_normal_answer_saved_and_displayed_with_sources():
    at = _run(ANSWER)
    assert not at.exception
    last = at.session_state["chat_history"][-1]
    assert last["role"] == "assistant"
    assert last["content"] == ANSWER
    assert last["sources"] == "Sources: page 2, 3"
    assert ANSWER in [m.value for m in at.markdown]
    assert "Sources: page 2, 3" in [c.value for c in at.caption]

    # re-render from history (next rerun) still shows text and sources
    at.run()
    assert ANSWER in [m.value for m in at.markdown]
    assert "Sources: page 2, 3" in [c.value for c in at.caption]


def test_refusal_has_no_sources():
    at = _run("I could not find this information in the document.")
    assert at.session_state["chat_history"][-1]["sources"] == ""
    assert not [c for c in at.caption if c.value.startswith("Sources:")]


def test_empty_model_answer_shows_message_and_no_sources():
    at = _run("   ")
    last = at.session_state["chat_history"][-1]
    assert last["content"].strip() != ""
    assert last["sources"] == ""


def test_translator_returning_none_falls_back_to_english_answer():
    from unittest import mock
    with mock.patch("deep_translator.GoogleTranslator.translate", return_value=None):
        at = AppTest.from_file("app.py", default_timeout=120)
        at.session_state["pdf_processed"] = True
        at.session_state["chat_history"] = []
        at.session_state["qa_chain"] = FakeChain(ANSWER)
        at.session_state["chunk_count"] = 1
        at.session_state["char_count"] = 1
        at.run()
        at.chat_input[0].set_value("किसानों को प्रति वर्ष कितना पैसा मिलता है?").run()
    last = at.session_state["chat_history"][-1]
    assert last["content"] == ANSWER
    assert last["sources"] == "Sources: page 2, 3"
    assert ANSWER in [m.value for m in at.markdown]
