"""Detect prompt artifacts in repository files."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List

from scan.repo_scanner import FileObject


@dataclass(frozen=True)
class DetectedPrompt:
    file_path: str
    line_numbers: List[int]
    prompt_type: str
    raw_text: str


SYSTEM_INDICATORS = [
    re.compile(r"\bsystem prompt\b", re.IGNORECASE),
    re.compile(r"^\s*system\s*:", re.IGNORECASE),
    re.compile(r"\byou are (a|an)\b", re.IGNORECASE),
    re.compile(r"\bact as\b", re.IGNORECASE),
]

USER_INDICATORS = [
    re.compile(r"\buser prompt\b", re.IGNORECASE),
    re.compile(r"^\s*user\s*:", re.IGNORECASE),
    re.compile(r"\buser input\b", re.IGNORECASE),
]

TEMPLATE_INDICATORS = [
    re.compile(r"\{user_input\}"),
    re.compile(r"\{input\}"),
    re.compile(r"\{prompt\}"),
    re.compile(r"\{query\}"),
    re.compile(r"\{\{[^}]+\}\}"),
    re.compile(r"\$\{[^}]+\}"),
    re.compile(r"f[\"'].*\{[^}]+\}.*[\"']"),
]

TRIPLE_QUOTE_PATTERN = re.compile(r"('''|\"\"\")(.*?)(\1)", re.DOTALL)


def detect_prompts(file_obj: FileObject) -> List[DetectedPrompt]:
    """Detect prompt artifacts from a FileObject."""
    detected: List[DetectedPrompt] = []
    covered_ranges: List[range] = []

    for match in TRIPLE_QUOTE_PATTERN.finditer(file_obj.content):
        block_text = match.group(0)
        prompt_type = _classify_prompt(block_text)
        if prompt_type is None:
            continue

        start_line = file_obj.content[: match.start()].count("\n") + 1
        end_line = start_line + block_text.count("\n")
        line_numbers = list(range(start_line, end_line + 1))
        covered_ranges.append(range(start_line, end_line + 1))
        detected.append(
            DetectedPrompt(
                file_path=file_obj.path,
                line_numbers=line_numbers,
                prompt_type=prompt_type,
                raw_text=block_text,
            )
        )

    for line_number, line in enumerate(file_obj.content.splitlines(), start=1):
        if _line_covered(line_number, covered_ranges):
            continue
        prompt_type = _classify_prompt(line)
        if prompt_type is None:
            continue
        detected.append(
            DetectedPrompt(
                file_path=file_obj.path,
                line_numbers=[line_number],
                prompt_type=prompt_type,
                raw_text=line,
            )
        )

    return detected


def _line_covered(line_number: int, ranges: Iterable[range]) -> bool:
    return any(line_number in line_range for line_range in ranges)


def _classify_prompt(text: str) -> str | None:
    if _matches_any(SYSTEM_INDICATORS, text):
        return "system"
    if _matches_any(USER_INDICATORS, text):
        return "user"
    if _matches_any(TEMPLATE_INDICATORS, text):
        return "template"
    return None


def _matches_any(patterns: Iterable[re.Pattern], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)
