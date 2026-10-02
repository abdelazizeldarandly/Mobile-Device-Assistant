import re
import unicodedata

# Regex patterns for common prompt-injection phrases.
# \s+ matches any run of whitespace, so "ignore   previous" is still caught.
SUSPICIOUS_PATTERNS = [
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)",
    r"disregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)",
    r"forget\s+(all\s+|your\s+|the\s+)?(previous|prior|instructions)",
    r"system\s+prompt",
    r"you\s+are\s+now",
    r"new\s+instructions",
    r"reveal\s+(your|the)\s+(prompt|instructions|rules)",
]

_COMPILED = [re.compile(p) for p in SUSPICIOUS_PATTERNS]


class PromptInjectionError(ValueError):
    """Raised when the input looks like a prompt-injection attempt."""


def _normalize(text: str) -> str:
    """Lowercase and strip tricks like zero-width characters or odd Unicode forms."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")  # zero-width chars
    return text.lower()


def check_prompt_injection(text: str) -> None:
    """Raise PromptInjectionError if the text contains a suspicious phrase."""
    normalized = _normalize(text)
    for pattern in _COMPILED:
        if pattern.search(normalized):
            raise PromptInjectionError("Input rejected: suspicious content detected.")