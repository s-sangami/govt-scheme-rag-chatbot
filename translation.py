"""Question/answer translation with a retry and a second backend.

Google (via deep-translator) rate-limits by IP and answers HTTP 429, which
deep-translator raises as TooManyRequests. When that happens we retry once,
then fall back to MyMemory. If every backend fails we raise TranslationError
carrying the real reasons, so the UI can say so instead of failing silently.
"""
import logging
import time

from deep_translator import GoogleTranslator, MyMemoryTranslator
from langdetect import detect

logger = logging.getLogger(__name__)

MYMEMORY_CODES = {
    "en": "en-US", "hi": "hi-IN", "ta": "ta-IN",
    "te": "te-IN", "kn": "kn-IN", "ml": "ml-IN",
}


class TranslationError(Exception):
    """All translation backends failed or returned nothing."""


def _google(text, source, target):
    return GoogleTranslator(source=source, target=target).translate(text)


def _mymemory(text, source, target):
    src = MYMEMORY_CODES.get(source, source)
    tgt = MYMEMORY_CODES.get(target, target)
    return MyMemoryTranslator(source=src, target=tgt).translate(text)


DEFAULT_BACKENDS = (_google, _mymemory)


def translate(text, source, target, backends=None, retry_delay=1.0):
    """Translate text; try each backend (the first one twice). Raises TranslationError."""
    errors = []
    for index, backend in enumerate(backends or DEFAULT_BACKENDS):
        for attempt in range(2 if index == 0 else 1):
            try:
                result = backend(text, source, target)
                if result and result.strip():
                    return result
                raise ValueError("empty translation")
            except Exception as exc:
                name = getattr(backend, "__name__", "translator")
                logger.warning("%s failed (%s): %s", name, type(exc).__name__, exc)
                errors.append(f"{name}: {type(exc).__name__}")
                if attempt == 0 and index == 0 and retry_delay:
                    time.sleep(retry_delay)
    raise TranslationError("; ".join(errors))


def translate_to_english(text, backends=None, retry_delay=1.0):
    """Returns (english_text, detected_lang, ok).

    On failure ok=False and english_text is the original text. detected_lang is
    kept when detection worked, so the caller can tell a failed translation of
    a non-English question from a question that is simply English.
    """
    try:
        if len(text.strip()) < 10:
            return text, "en", True
        lang = detect(text)
    except Exception:
        logger.exception("Language detection failed")
        return text, "en", False
    if lang == "en":
        return text, "en", True
    try:
        return translate(text, lang, "en", backends, retry_delay), lang, True
    except TranslationError:
        logger.exception("Translation to English failed")
        return text, lang, False


def translate_from_english(text, target_lang, backends=None, retry_delay=1.0):
    """Returns (translated_text, ok). On failure returns the English text."""
    if target_lang == "en":
        return text, True
    try:
        return translate(text, "en", target_lang, backends, retry_delay), True
    except TranslationError:
        logger.exception("Translation from English to %s failed", target_lang)
        return text, False
