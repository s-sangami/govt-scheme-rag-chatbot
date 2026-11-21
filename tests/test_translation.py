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


# ---- chunked, structure-preserving answer translation ----

TABLE = (
    "**Eligibility criteria**\n"
    "| Criterion | Details |\n"
    "|-----------|---------|\n"
    "| **Land status** | Name in land records.<br>Cut-off 1 Feb 2019. |\n"
    "- Farmers only\n"
    "1. Apply online"
)


class Upper:
    """Fake translator: upper-cases and records every request."""

    def __init__(self):
        self.seen = []

    def __call__(self, text, source, target):
        self.seen.append(text)
        return text.upper()


def run(text, *backends):
    return tr.translate_from_english(text, "hi", backends=backends or (Upper(),), chunk_delay=0)


def test_table_structure_is_preserved():
    out, ok = run(TABLE)
    assert ok
    assert out == (
        "**ELIGIBILITY CRITERIA**\n"
        "| CRITERION | DETAILS |\n"
        "|-----------|---------|\n"
        "| **LAND STATUS** | NAME IN LAND RECORDS.<br>CUT-OFF 1 FEB 2019. |\n"
        "- FARMERS ONLY\n"
        "1. APPLY ONLINE"
    )


def test_long_answer_is_split_into_chunks_under_400():
    fake = Upper()
    sentence = "Farmers with land can apply for the scheme. "
    answer = (sentence * 30).strip() + "\n\n" + (sentence * 20).strip()
    out, ok = tr.translate_from_english(answer, "hi", backends=(fake,), chunk_delay=0)
    assert ok and len(fake.seen) > 3
    assert all(len(c) <= 400 for c in fake.seen)
    assert all(c.endswith(".") for c in fake.seen)  # never mid-sentence
    assert out.count("\n\n") == 1 and out == out.upper()


def test_one_failing_chunk_stays_english_others_translated():
    def picky(text, source, target):
        if "Cut-off" in text:
            raise TooManyRequests()
        return text.upper()

    out, ok = run(TABLE, picky)
    assert ok is False
    assert "CUT-OFF" not in out and "Cut-off 1 Feb 2019." in out
    assert "NAME IN LAND RECORDS." in out and "| **LAND STATUS** |" in out


def test_both_services_failing_returns_english_and_not_ok():
    assert run(TABLE, rate_limited, rate_limited) == (TABLE, False)


def test_short_answer_is_one_plain_request():
    fake = Upper()
    assert tr.translate_from_english("Rs. 6,000 per year", "hi", backends=(fake,)) == ("RS. 6,000 PER YEAR", True)
    assert fake.seen == ["Rs. 6,000 per year"]


def test_each_chunk_retries_then_uses_second_service():
    calls = {"a": 0, "b": 0}

    def a(text, s, t):
        calls["a"] += 1
        raise TooManyRequests()

    def b(text, s, t):
        calls["b"] += 1
        return "X"

    out, ok = run("Apply online.\nPay later.", a, b)
    assert ok and out == "X\nX"
    assert calls == {"a": 4, "b": 2}  # 2 tries on the first service per chunk, then the fallback
