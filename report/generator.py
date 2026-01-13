"""Generate governance reports from AI applications and policy violations."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence

from governance.objects import AIApplication, DataSource, Model
from policy.engine import PolicyViolation


def generate_report(
    ai_applications: Sequence[AIApplication],
    violations: Sequence[PolicyViolation],
) -> Dict[str, Any]:
    """Generate a JSON-serializable governance report."""
    inventory = [_serialize_application(app) for app in ai_applications]
    risk_levels = _summarize_risk_levels(ai_applications)
    policy_violations = [_serialize_violation(violation) for violation in violations]
    executive_summary = _build_executive_summary(ai_applications, violations, risk_levels)

    return {
        "executive_summary": executive_summary,
        "ai_inventory": inventory,
        "risk_levels": risk_levels,
        "policy_violations": policy_violations,
    }


def generate_markdown_summary(report: Dict[str, Any]) -> str:
    """Generate a human-readable Markdown summary from a report."""
    summary = report.get("executive_summary", {})
    inventory = report.get("ai_inventory", [])
    risk_levels = report.get("risk_levels", {})
    policy_violations = report.get("policy_violations", [])

    lines = ["# Governance Report", ""]
    lines.extend(_format_executive_summary(summary))
    lines.extend(_format_inventory(inventory))
    lines.extend(_format_risk_levels(risk_levels))
    lines.extend(_format_policy_violations(policy_violations))

    return "\n".join(lines).strip() + "\n"


def _serialize_application(app: AIApplication) -> Dict[str, Any]:
    return {
        "name": app.name,
        "owner": app.owner,
        "risk_level": app.risk_level,
        "models": [_serialize_model(model) for model in app.models],
        "data_sources": [_serialize_data_source(source) for source in app.data_sources],
        "capabilities": list(app.capabilities),
        "policies": list(app.policies),
    }


def _serialize_model(model: Model) -> Dict[str, str]:
    return {
        "provider": model.provider,
        "model_class": model.model_class,
        "risk_tier": model.risk_tier,
    }


def _serialize_data_source(source: DataSource) -> Dict[str, str]:
    return {
        "type": source.type,
        "sensitivity": source.sensitivity,
    }


def _serialize_violation(violation: PolicyViolation) -> Dict[str, str]:
    return {
        "policy_id": violation.policy_id,
        "severity": violation.severity,
        "ai_application": violation.ai_app.name,
        "reason": violation.reason,
    }


def _summarize_risk_levels(ai_applications: Sequence[AIApplication]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for app in ai_applications:
        grouped[app.risk_level].append(app.name)
    return {level: sorted(names) for level, names in grouped.items()}


def _build_executive_summary(
    ai_applications: Sequence[AIApplication],
    violations: Sequence[PolicyViolation],
    risk_levels: Dict[str, List[str]],
) -> Dict[str, Any]:
    risk_counts = Counter(app.risk_level for app in ai_applications)
    highest_risk = _highest_risk_applications(ai_applications)
    return {
        "total_applications": len(ai_applications),
        "risk_level_counts": dict(risk_counts),
        "total_policy_violations": len(violations),
        "highest_risk_applications": highest_risk,
        "policy_violations_by_severity": _violation_counts_by_severity(violations),
        "risk_levels_in_scope": sorted(risk_levels.keys()),
    }


def _highest_risk_applications(ai_applications: Sequence[AIApplication]) -> List[str]:
    order = {"high": 3, "medium": 2, "low": 1}
    highest_score = 0
    highest: List[str] = []
    for app in ai_applications:
        score = order.get(app.risk_level, 0)
        if score > highest_score:
            highest_score = score
            highest = [app.name]
        elif score == highest_score and score > 0:
            highest.append(app.name)
    return sorted(highest)


def _violation_counts_by_severity(violations: Sequence[PolicyViolation]) -> Dict[str, int]:
    return dict(Counter(violation.severity for violation in violations))


def _format_executive_summary(summary: Dict[str, Any]) -> List[str]:
    lines = ["## Executive summary", ""]
    total_apps = summary.get("total_applications", 0)
    total_violations = summary.get("total_policy_violations", 0)
    lines.append(f"- Total AI applications: {total_apps}")

    risk_counts = summary.get("risk_level_counts", {})
    if risk_counts:
        risk_parts = ", ".join(f"{level}: {count}" for level, count in sorted(risk_counts.items()))
        lines.append(f"- Risk levels: {risk_parts}")
    else:
        lines.append("- Risk levels: none")

    lines.append(f"- Policy violations: {total_violations}")
    highest = summary.get("highest_risk_applications", [])
    if highest:
        lines.append(f"- Highest risk applications: {', '.join(highest)}")
    else:
        lines.append("- Highest risk applications: none")
    lines.append("")
    return lines


def _format_inventory(inventory: Iterable[Dict[str, Any]]) -> List[str]:
    lines = ["## AI inventory", ""]
    header = "| Application | Owner | Risk level | Models | Data sources | Capabilities | Policies |"
    separator = "| --- | --- | --- | --- | --- | --- | --- |"
    lines.extend([header, separator])

    items = list(inventory)
    if not items:
        lines.append("| None | - | - | - | - | - | - |")
        lines.append("")
        return lines

    for item in items:
        models = _format_models(item.get("models", []))
        data_sources = _format_data_sources(item.get("data_sources", []))
        capabilities = _format_list(item.get("capabilities", []))
        policies = _format_list(item.get("policies", []))
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_md(item.get("name", "unknown")),
                    _escape_md(item.get("owner", "unknown")),
                    _escape_md(item.get("risk_level", "unknown")),
                    _escape_md(models),
                    _escape_md(data_sources),
                    _escape_md(capabilities),
                    _escape_md(policies),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _format_risk_levels(risk_levels: Dict[str, List[str]]) -> List[str]:
    lines = ["## Risk levels", ""]
    if not risk_levels:
        lines.append("- No applications detected.")
        lines.append("")
        return lines
    for level in sorted(risk_levels.keys()):
        apps = risk_levels[level]
        if apps:
            lines.append(f"- **{level.capitalize()}**: {', '.join(sorted(apps))}")
        else:
            lines.append(f"- **{level.capitalize()}**: none")
    lines.append("")
    return lines


def _format_policy_violations(violations: Iterable[Dict[str, Any]]) -> List[str]:
    lines = ["## Policy violations", ""]
    violations_list = list(violations)
    if not violations_list:
        lines.append("No policy violations detected.")
        lines.append("")
        return lines

    header = "| Policy ID | Severity | Application | Reason |"
    separator = "| --- | --- | --- | --- |"
    lines.extend([header, separator])
    for violation in violations_list:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_md(violation.get("policy_id", "")),
                    _escape_md(violation.get("severity", "")),
                    _escape_md(violation.get("ai_application", "")),
                    _escape_md(violation.get("reason", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _format_models(models: Iterable[Dict[str, str]]) -> str:
    parts = []
    for model in models:
        provider = model.get("provider", "unknown")
        model_class = model.get("model_class", "unknown")
        risk_tier = model.get("risk_tier", "unknown")
        parts.append(f"{provider}:{model_class} ({risk_tier})")
    return _format_list(parts)


def _format_data_sources(sources: Iterable[Dict[str, str]]) -> str:
    parts = []
    for source in sources:
        source_type = source.get("type", "unknown")
        sensitivity = source.get("sensitivity", "unknown")
        parts.append(f"{source_type} ({sensitivity})")
    return _format_list(parts)


def _format_list(values: Iterable[str]) -> str:
    items = [value for value in values if value]
    return ", ".join(items) if items else "none"


def _escape_md(value: Any) -> str:
    text = str(value) if value is not None else ""
    return text.replace("|", "\\|")
