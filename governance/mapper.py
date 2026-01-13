"""Governance mapper for detected AI usage signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from governance.objects import AIApplication, DataSource, Model
from scan.detectors.llm_client_detector import LLMClientUsage
from scan.detectors.prompt_detector import DetectedPrompt
from scan.detectors.rag_detector import RAGPipeline


@dataclass
class _AppAggregation:
    prompts: List[DetectedPrompt] = field(default_factory=list)
    llm_clients: List[LLMClientUsage] = field(default_factory=list)
    rag_pipelines: List[RAGPipeline] = field(default_factory=list)


def map_to_ai_applications(
    detected_prompts: Sequence[DetectedPrompt],
    llm_client_usages: Sequence[LLMClientUsage],
    rag_pipelines: Sequence[RAGPipeline],
) -> List[AIApplication]:
    """Map detected signals into canonical AIApplication objects."""
    aggregations: Dict[str, _AppAggregation] = {}

    _add_prompts(aggregations, detected_prompts)
    _add_llm_clients(aggregations, llm_client_usages)
    _add_rag_pipelines(aggregations, rag_pipelines)

    applications: List[AIApplication] = []
    for app_key in sorted(aggregations):
        aggregation = aggregations[app_key]
        applications.append(_build_application(app_key, aggregation))

    return applications


def _add_prompts(
    aggregations: Dict[str, _AppAggregation],
    prompts: Sequence[DetectedPrompt],
) -> None:
    for prompt in prompts:
        aggregation = aggregations.setdefault(_app_key(prompt.file_path), _AppAggregation())
        aggregation.prompts.append(prompt)


def _add_llm_clients(
    aggregations: Dict[str, _AppAggregation],
    llm_clients: Sequence[LLMClientUsage],
) -> None:
    for usage in llm_clients:
        aggregation = aggregations.setdefault(_app_key(usage.file_path), _AppAggregation())
        aggregation.llm_clients.append(usage)


def _add_rag_pipelines(
    aggregations: Dict[str, _AppAggregation],
    rag_pipelines: Sequence[RAGPipeline],
) -> None:
    for pipeline in rag_pipelines:
        aggregation = aggregations.setdefault(_app_key(pipeline.file_path), _AppAggregation())
        aggregation.rag_pipelines.append(pipeline)


def _app_key(file_path: Optional[str]) -> str:
    if not file_path:
        return "unknown"
    normalized = file_path.lstrip("./")
    if not normalized:
        return "unknown"
    return normalized.split("/", maxsplit=1)[0] or "unknown"


def _build_application(app_key: str, aggregation: _AppAggregation) -> AIApplication:
    models = _collect_models(aggregation.llm_clients, aggregation.rag_pipelines)
    data_sources = _collect_data_sources(aggregation.rag_pipelines)
    capabilities = _collect_capabilities(aggregation)
    risk_level = _infer_risk_level(aggregation)

    return AIApplication(
        name=app_key or "unknown",
        owner="unknown",
        risk_level=risk_level,
        models=models,
        data_sources=data_sources,
        capabilities=capabilities,
        policies=["unknown"],
    )


def _collect_models(
    llm_clients: Sequence[LLMClientUsage],
    rag_pipelines: Sequence[RAGPipeline],
) -> List[Model]:
    models: List[Model] = []

    for usage in llm_clients:
        model_class = usage.model_name or "unknown"
        models.append(
            Model(
                provider=usage.provider or "unknown",
                model_class=model_class,
                risk_tier="unknown",
            )
        )

    for pipeline in rag_pipelines:
        if not pipeline.embedding_model:
            continue
        models.append(
            Model(
                provider="unknown",
                model_class=pipeline.embedding_model,
                risk_tier="unknown",
            )
        )

    return _dedupe_models(models)


def _collect_data_sources(rag_pipelines: Sequence[RAGPipeline]) -> List[DataSource]:
    sources: List[DataSource] = []

    for pipeline in rag_pipelines:
        source_type = pipeline.data_source_hint or pipeline.vector_store_type or "unknown"
        sources.append(
            DataSource(
                type=source_type,
                sensitivity="unknown",
            )
        )

    return _dedupe_data_sources(sources)


def _collect_capabilities(aggregation: _AppAggregation) -> List[str]:
    capabilities: Set[str] = set()
    if aggregation.prompts:
        capabilities.add("prompting")
    if aggregation.llm_clients:
        capabilities.add("generation")
    if aggregation.rag_pipelines:
        capabilities.add("retrieval_augmented_generation")
    return sorted(capabilities)


def _infer_risk_level(aggregation: _AppAggregation) -> str:
    if _has_sensitive_sources(aggregation.rag_pipelines):
        return "high"
    if aggregation.rag_pipelines:
        return "medium"
    if aggregation.llm_clients:
        return "medium"
    if aggregation.prompts:
        return "low"
    return "low"


def _has_sensitive_sources(rag_pipelines: Sequence[RAGPipeline]) -> bool:
    for pipeline in rag_pipelines:
        hint = pipeline.data_source_hint or ""
        if _looks_external_source(hint):
            return True
    return False


def _looks_external_source(hint: str) -> bool:
    lowered = hint.lower()
    return any(
        marker in lowered
        for marker in (
            "http://",
            "https://",
            "s3://",
            "gs://",
            ".pdf",
            ".csv",
            ".txt",
            ".md",
            ".docx",
            ".json",
            ".html",
            ".xlsx",
        )
    )


def _dedupe_models(models: Iterable[Model]) -> List[Model]:
    seen: Set[Tuple[str, str, str]] = set()
    unique: List[Model] = []
    for model in models:
        key = (model.provider, model.model_class, model.risk_tier)
        if key in seen:
            continue
        seen.add(key)
        unique.append(model)
    return sorted(unique, key=lambda item: (item.provider, item.model_class, item.risk_tier))


def _dedupe_data_sources(sources: Iterable[DataSource]) -> List[DataSource]:
    seen: Set[Tuple[str, str]] = set()
    unique: List[DataSource] = []
    for source in sources:
        key = (source.type, source.sensitivity)
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return sorted(unique, key=lambda item: (item.type, item.sensitivity))
