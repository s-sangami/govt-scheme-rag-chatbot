import pytest
from deep_translator.exceptions import TooManyRequests

import translation as tr

HINDI = "इस योजना के लिए कौन पात्र है?"
TAMIL = "யார் தகுதியானவர்கள்?"


def ok(text, source, target):
    return "Who is eligible for this scheme?"


def rate_limited(text, source, target):
    raise TooManyRequests()


def returns_none(text, source, target):
    return None


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(tr.time, "sleep", lambda s: None)


@pytest.mark.parametrize("question,lang", [(HINDI, "hi"), (TAMIL, "ta")])
def test_success(question, lang):
    assert tr.translate_to_english(question, backends=(ok,)) == (
        "Who is eligible for this scheme?", lang, True)


def test_english_is_not_translated():
    q = "Who is eligible for PM Kisan scheme?"
    assert tr.translate_to_english(q, backends=(rate_limited,)) == (q, "en", True)


def test_failure_keeps_detected_language_and_reports_not_ok():
    assert tr.translate_to_english(HINDI, backends=(rate_limited, rate_limited)) == (HINDI, "hi", False)


def test_none_result_counts_as_failure():
    assert tr.translate_to_english(HINDI, backends=(returns_none,)) == (HINDI, "hi", False)


def test_falls_back_to_second_backend_when_first_is_rate_limited():
    assert tr.translate_to_english(HINDI, backends=(rate_limited, ok))[2] is True


def test_first_backend_is_retried_once():
    calls = []

    def flaky(text, source, target):
        calls.append(1)
        if len(calls) == 1:
            raise TooManyRequests()
        return "Who is eligible?"

    assert tr.translate(HINDI, "auto", "en", backends=(flaky,)) == "Who is eligible?"
    assert len(calls) == 2


def test_error_names_the_real_exception():
    with pytest.raises(tr.TranslationError, match="TooManyRequests"):
        tr.translate(HINDI, "auto", "en", backends=(rate_limited,))


def test_answer_back_translation_success_failure_and_none():
    assert tr.translate_from_english("Rs. 6,000", "hi", backends=(lambda *a: "छह हजार",)) == ("छह हजार", True)
    assert tr.translate_from_english("Rs. 6,000", "hi", backends=(rate_limited,)) == ("Rs. 6,000", False)
    assert tr.translate_from_english("Rs. 6,000", "ta", backends=(returns_none,)) == ("Rs. 6,000", False)
    assert tr.translate_from_english("Rs. 6,000", "en", backends=(rate_limited,)) == ("Rs. 6,000", True)


def test_detected_language_is_passed_as_source_not_auto():
    seen = []

    def spy(text, source, target):
        seen.append((source, target))
        return "ok"

    tr.translate_to_english(TAMIL, backends=(spy,))
    assert seen == [("ta", "en")]
