"""Shared utils

Provides common functionality for:
- Extracting data from events
- Pattern matching
- Field comparison operations
"""

import re


def matches_pattern(
    text: str,
    pattern: str,
    case_sensitive: bool = False,
    is_regex: bool = False,
) -> tuple[bool, str | None]:
    """Check if text matches a pattern.

    Args:
        text: Text to search in
        pattern: Pattern to match (string or regex)
        case_sensitive: Whether matching is case-sensitive
        is_regex: Whether pattern is a regex

    Returns:
        Tuple of (matched: bool, matched_text: str | None)
        matched_text contains the actual matched substring for evidence
    """
    if not text or not pattern:
        return False, None

    if is_regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            match = re.search(pattern, text, flags)
            if match:
                return True, match.group(0)
        except re.error:
            # Invalid regex, fall back to literal match
            pass

    # Literal string matching
    search_text = text if case_sensitive else text.lower()
    search_pattern = pattern if case_sensitive else pattern.lower()

    if search_pattern in search_text:
        # Find actual matched text for evidence
        start = search_text.find(search_pattern)
        matched = text[start : start + len(pattern)]
        return True, matched

    return False, None


def extract_context(
    text: str, match_start: int, match_length: int, context_chars: int = 50
) -> str:
    """Extract context around a match for evidence.

    Args:
        text: Full text
        match_start: Start index of match
        match_length: Length of match
        context_chars: Characters of context on each side

    Returns:
        Context string with ellipsis if truncated
    """
    start = max(0, match_start - context_chars)
    end = min(len(text), match_start + match_length + context_chars)

    context = text[start:end]

    if start > 0:
        context = "..." + context
    if end < len(text):
        context = context + "..."

    return context
