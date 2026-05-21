"""Safe display-only cleanup for noisy OCR-ish labels and excerpts."""

from __future__ import annotations

import re

SAFE_TOKEN_REPLACEMENTS = {
    "auih": "Allah",
    "au&": "Allah",
    "im8m": "Imam",
    "qtblah": "Qiblah",
    "wth": "with",
}


def clean_display_label(value: str) -> str:
    text = _normalize_whitespace(_apply_safe_replacements(value))
    if not text:
        return ""

    original = text
    cleaned_tokens: list[str] = []
    for token in text.split(" "):
        cleaned = _trim_punctuation(token)
        if not cleaned:
            continue
        if _looks_like_garbled_label_token(cleaned):
            continue
        cleaned_tokens.append(cleaned)

    collapsed = _normalize_whitespace(" ".join(cleaned_tokens))
    if not collapsed:
        return original
    if len(collapsed) < max(8, int(len(original) * 0.45)):
        return original
    return collapsed


def clean_display_excerpt(value: str) -> str:
    text = _normalize_whitespace(_apply_safe_replacements(value))
    if not text:
        return ""

    original = text
    cleaned_tokens: list[str] = []
    for token in text.split(" "):
        cleaned = _trim_punctuation(token)
        if not cleaned:
            continue
        if _looks_like_garbled_excerpt_token(cleaned):
            continue
        cleaned_tokens.append(cleaned)

    collapsed = _normalize_whitespace(" ".join(cleaned_tokens))
    if not collapsed:
        return original
    if len(collapsed) < max(12, int(len(original) * 0.6)):
        return original
    return collapsed


def _apply_safe_replacements(value: str) -> str:
    text = str(value or "")
    for source, target in SAFE_TOKEN_REPLACEMENTS.items():
        text = re.sub(rf"\b{re.escape(source)}\b", target, text, flags=re.IGNORECASE)
    return text


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _trim_punctuation(token: str) -> str:
    return token.strip("[]{}()<>\"'`.,;:!?|")


def _looks_like_garbled_label_token(token: str) -> bool:
    if len(token) <= 2:
        return False
    if _has_mixed_letters_and_digits(token):
        return True
    symbol_count = sum(1 for char in token if not char.isalnum() and char not in {"-", "'"})
    if symbol_count >= 2:
        return True
    if len(token) >= 5 and not any(char in "aeiouAEIOU" for char in token) and any(
        char.isalpha() for char in token
    ):
        return True
    return False


def _looks_like_garbled_excerpt_token(token: str) -> bool:
    if len(token) <= 3:
        return False
    if _has_mixed_letters_and_digits(token):
        return True
    symbol_count = sum(1 for char in token if not char.isalnum() and char not in {"-", "'"})
    return symbol_count >= 3


def _has_mixed_letters_and_digits(token: str) -> bool:
    has_alpha = any(char.isalpha() for char in token)
    has_digit = any(char.isdigit() for char in token)
    return has_alpha and has_digit
