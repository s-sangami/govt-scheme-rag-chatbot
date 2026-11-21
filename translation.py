"""Question/answer translation with a retry and a second backend.

Google (via deep-translator) rate-limits by IP and answers HTTP 429, which
deep-translator raises as TooManyRequests. When that happens we retry once,
then fall back to MyMemory. If every backend fails we raise TranslationError
carrying the real reasons, so the UI can say so instead of failing silently.
"""
import logging
import re
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


MAX_CHUNK = 400
_LETTER = re.compile(r"[^\W\d_]")
_BR = re.compile(r"(<br\s*/?>)", re.I)
_PREFIX = re.compile(r"^(\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s+)?)(.*)$", re.S)
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")


def _split_long(text, limit=MAX_CHUNK):
    """Split text into pieces <= limit, on sentence boundaries (words as a last resort)."""
    if len(text) <= limit:
        return [text]
    pieces, current = [], ""
    for sentence in _SENTENCE_END.split(text):
        while len(sentence) > limit:  # a single huge sentence: break on spaces
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            head, sentence = sentence[:cut], sentence[cut:].lstrip()
            if current:
                pieces.append(current)
                current = ""
            pieces.append(head)
        if current and len(current) + 1 + len(sentence) > limit:
            pieces.append(current)
            current = ""
        current = f"{current} {sentence}".strip() if current else sentence
    if current:
        pieces.append(current)
    return pieces


class _Chunker:
    """Translates text segment by segment, keeping a segment in English if it fails."""

    def __init__(self, source, target, backends, retry_delay, chunk_delay):
        self.args = (source, target, backends, retry_delay)
        self.chunk_delay = chunk_delay
        self.calls = 0
        self.failed = 0

    def segment(self, text):
        """Translate one piece of plain text (<= MAX_CHUNK chars)."""
        if not _LETTER.search(text):
            return text
        if self.calls and self.chunk_delay:
            time.sleep(self.chunk_delay)
        self.calls += 1
        source, target, backends, retry_delay = self.args
        try:
            return translate(text, source, target, backends, retry_delay).strip()
        except TranslationError:
            logger.warning("Chunk kept in English (%d chars)", len(text))
            self.failed += 1
            return text

    def text(self, text):
        """Translate text that may be long, preserving leading/trailing whitespace and **bold** wrapping."""
        core = text.strip()
        if not core:
            return text
        lead, trail = text[: len(text) - len(text.lstrip())], text[len(text.rstrip()):]
        bold = core.startswith("**") and core.endswith("**") and len(core) > 4 and "**" not in core[2:-2]
        if bold:
            core = core[2:-2].strip()
        out = " ".join(self.segment(p) for p in _split_long(core))
        return f"{lead}**{out}**{trail}" if bold else f"{lead}{out}{trail}"

    def with_br(self, text):
        return "".join(part if _BR.fullmatch(part) else self.text(part) for part in _BR.split(text))

    def line(self, line):
        if line.lstrip().startswith("|"):
            cells = line.split("|")
            return "|".join(self.with_br(c) if 0 < i < len(cells) - (1 if line.rstrip().endswith("|") else 0) else c
                            for i, c in enumerate(cells))
        prefix, body = _PREFIX.match(line).groups()
        return prefix + self.with_br(body)


def translate_markdown(text, source, target, backends=None, retry_delay=1.0, chunk_delay=0.3):
    """Translate a markdown answer chunk by chunk. Returns (text, failed_chunks).

    Table pipes, headings, bullets and <br> tags stay as they are; only the
    text between them is translated. A chunk that fails stays in English.
    """
    chunker = _Chunker(source, target, backends, retry_delay, chunk_delay)
    out = "\n".join(chunker.line(line) for line in text.split("\n"))
    return out, chunker.failed


def translate_from_english(text, target_lang, backends=None, retry_delay=1.0, chunk_delay=0.3):
    """Returns (translated_text, ok).

    Long/markdown answers are translated in chunks. ok is False if any chunk
    fell back to English (the text is then partly or wholly English).
    """
    if target_lang == "en":
        return text, True
    try:
        result, failed = translate_markdown(text, "en", target_lang, backends, retry_delay, chunk_delay)
    except Exception:
        logger.exception("Translation from English to %s failed", target_lang)
        return text, False
    return result, failed == 0
