"""Detect LLM client usage in repository files."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List, Optional, Sequence

from scan.repo_scanner import FileObject


@dataclass(frozen=True)
class LLMClientUsage:
    file_path: str
    line_numbers: List[int]
    provider: str
    model_name: Optional[str]


MODEL_PATTERNS = [
    re.compile(r"\bmodel\s*[:=]\s*[\"']([^\"']+)[\"']"),
    re.compile(r"\bmodel_name\s*[:=]\s*[\"']([^\"']+)[\"']"),
    re.compile(r"\bdeployment_name\s*[:=]\s*[\"']([^\"']+)[\"']"),
    re.compile(r"\bazure_deployment\s*[:=]\s*[\"']([^\"']+)[\"']"),
    re.compile(r"\bengine\s*[:=]\s*[\"']([^\"']+)[\"']"),
]


PROVIDER_PATTERNS = [
    {
        "provider": "OpenAI",
        "patterns": [
            r"\bfrom\s+openai\s+import\s+OpenAI\b",
            r"\bOpenAI\s*\(",
            r"\bimport\s+OpenAI\b.*\bfrom\s+[\"']openai[\"']",
            r"\bnew\s+OpenAI\s*\(",
            r"\bopenai\.chat\.completions\.create\s*\(",
            r"\bopenai\.completions\.create\s*\(",
            r"\bopenai\.responses\.create\s*\(",
        ],
    },
    {
        "provider": "Azure OpenAI",
        "patterns": [
            r"\bfrom\s+openai\s+import\s+AzureOpenAI\b",
            r"\bopenai\.AzureOpenAI\s*\(",
            r"\bAzureOpenAI\s*\(",
            r"\bimport\s+\{?\s*AzureOpenAI\s*\}?\s+from\s+[\"']openai[\"']",
            r"\bnew\s+AzureOpenAI\s*\(",
        ],
    },
    {
        "provider": "Anthropic",
        "patterns": [
            r"\bimport\s+anthropic\b",
            r"\bfrom\s+anthropic\s+import\s+Anthropic\b",
            r"\bAnthropic\s*\(",
            r"\banthropic\.Client\s*\(",
            r"\bimport\s+Anthropic\b.*\bfrom\s+[\"']@anthropic-ai/sdk[\"']",
            r"\bnew\s+Anthropic\s*\(",
        ],
    },
    {
        "provider": "LangChain",
        "patterns": [
            r"\bimport\s+langchain\b",
            r"\bfrom\s+langchain\b",
            r"\bfrom\s+langchain_openai\b",
            r"[\"']langchain(?:/[^\"']*)?[\"']",
        ],
    },
    {
        "provider": "LlamaIndex",
        "patterns": [
            r"\bimport\s+llama_index\b",
            r"\bfrom\s+llama_index\b",
            r"\bllama_index\.",
            r"[\"']llamaindex[\"']",
        ],
    },
    {
        "provider": "Semantic Kernel",
        "patterns": [
            r"\bimport\s+semantic_kernel\b",
            r"\bfrom\s+semantic_kernel\b",
            r"\bsemantic_kernel\.",
            r"[\"']semantic-kernel[\"']",
        ],
    },
]


def detect_llm_clients(file_obj: FileObject) -> List[LLMClientUsage]:
    """Detect LLM client usage from a FileObject."""
    detections: List[LLMClientUsage] = []
    lines = file_obj.content.splitlines()

    for provider_entry in PROVIDER_PATTERNS:
        provider = provider_entry["provider"]
        for pattern in provider_entry["patterns"]:
            compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for match in compiled.finditer(file_obj.content):
                line_numbers = _line_numbers_for_span(file_obj.content, match.start(), match.end())
                model_name = _extract_model_name(lines, line_numbers, MODEL_PATTERNS)
                detections.append(
                    LLMClientUsage(
                        file_path=file_obj.path,
                        line_numbers=line_numbers,
                        provider=provider,
                        model_name=model_name,
                    )
                )

    return _dedupe_detections(detections)


def _line_numbers_for_span(content: str, start: int, end: int) -> List[int]:
    start_line = content[:start].count("\n") + 1
    end_line = content[:end].count("\n") + 1
    return list(range(start_line, end_line + 1))


def _extract_model_name(
    lines: Sequence[str],
    line_numbers: Sequence[int],
    patterns: Iterable[re.Pattern],
) -> Optional[str]:
    if not line_numbers:
        return None

    start_index = max(line_numbers[0] - 1, 0)
    end_index = min(start_index + 4, len(lines))
    window = "\n".join(lines[start_index:end_index])

    for pattern in patterns:
        match = pattern.search(window)
        if match:
            return match.group(1).strip()

    return None


def _dedupe_detections(detections: Iterable[LLMClientUsage]) -> List[LLMClientUsage]:
    seen = set()
    unique: List[LLMClientUsage] = []
    for detection in detections:
        key = (
            detection.file_path,
            tuple(detection.line_numbers),
            detection.provider,
            detection.model_name,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(detection)
    return unique
