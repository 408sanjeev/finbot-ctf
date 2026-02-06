"""Pattern Match Detector

Generic regex/keyword matching on event fields.
The foundational primitive for text-based detection.
"""

import logging
from typing import Any

from sqlalchemy.orm import Session

from finbot.ctf.detectors.base import BaseDetector
from finbot.ctf.detectors.registry import register_detector
from finbot.ctf.detectors.result import DetectionResult
from finbot.ctf.detectors.utils import extract_context, matches_pattern

logger = logging.getLogger(__name__)


@register_detector("PatternMatchDetector")
class PatternMatchDetector(BaseDetector):
    """Matches patterns (keywords or regex) against event field values.

    Configuration:
        field: str - Event field to check (required)
        patterns: list - Patterns to match (required)
            Each pattern can be:
            - A string (literal match)
            - A dict with 'regex' key for regex patterns
        match_mode: str - "any" (default) or "all"
        case_sensitive: bool - Default False
        min_matches: int - Minimum matches required (default 1)

    Example YAML:
        detector_class: PatternMatchDetector
        detector_config:
          field: "response_content"
          patterns:
            - "system prompt"
            - "you are a"
            - regex: "(?i)instructions?:"
          match_mode: "any"
    """

    def _validate_config(self) -> None:
        if "field" not in self.config:
            raise ValueError("PatternMatchDetector requires 'field' config")
        if "patterns" not in self.config:
            raise ValueError("PatternMatchDetector requires 'patterns' config")
        if not isinstance(self.config["patterns"], list):
            raise ValueError("'patterns' must be a list")
        if not self.config["patterns"]:
            raise ValueError("'patterns' cannot be empty")

        match_mode = self.config.get("match_mode", "any")
        if match_mode not in ("any", "all"):
            raise ValueError("match_mode must be 'any' or 'all'")

    def get_relevant_event_types(self) -> list[str]:
        """Override in subclass or configure via YAML.

        Default: match all LLM response events.
        """
        return self.config.get("event_types", ["agent.*.llm_request_success"])

    def check_event(self, event: dict[str, Any], db: Session) -> DetectionResult:
        """Check if event field matches configured patterns.
        Only needs the current event, db is unused.
        """

        field = self.config["field"]
        patterns = self.config["patterns"]
        match_mode = self.config.get("match_mode", "any")
        case_sensitive = self.config.get("case_sensitive", False)
        min_matches = self.config.get("min_matches", 1)

        # Extract field value from event
        field_value = event.get(field)

        if field_value is None:
            return DetectionResult(
                detected=False,
                message=f"Field '{field}' not found in event",
            )

        if not isinstance(field_value, str):
            field_value = str(field_value)

        # Check patterns
        matches = []
        for pattern_config in patterns:
            pattern, is_regex = self._parse_pattern(pattern_config)

            matched, matched_text = matches_pattern(
                field_value, pattern, case_sensitive, is_regex
            )

            if matched and matched_text:
                match_start = field_value.lower().find(matched_text.lower())
                context = extract_context(field_value, match_start, len(matched_text))
                matches.append(
                    {
                        "pattern": pattern,
                        "matched": matched_text,
                        "context": context,
                        "is_regex": is_regex,
                    }
                )

        # Evaluate results based on match_mode
        if match_mode == "all":
            detected = len(matches) == len(patterns)
        else:  # "any"
            detected = len(matches) >= min_matches

        if not detected:
            return DetectionResult(
                detected=False,
                confidence=len(matches) / len(patterns) if patterns else 0,
                message=f"Matched {len(matches)}/{len(patterns)} patterns (need {min_matches})",
                evidence={"matches": matches} if matches else {},
            )

        return DetectionResult(
            detected=True,
            confidence=min(1.0, len(matches) / len(patterns) + 0.2),
            message=f"Pattern match: {len(matches)} pattern(s) matched in '{field}'",
            evidence={
                "field": field,
                "matches": matches,
                "total_patterns": len(patterns),
            },
        )

    def _parse_pattern(self, pattern_config: str | dict) -> tuple[str, bool]:
        """Parse pattern config into (pattern, is_regex) tuple."""
        if isinstance(pattern_config, dict):
            if "regex" in pattern_config:
                return pattern_config["regex"], True
            # Fallback: treat first value as literal
            return str(list(pattern_config.values())[0]), False

        return str(pattern_config), False
