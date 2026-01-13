"""Policy engine for evaluating AI applications against YAML-defined rules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import yaml

from governance.objects import AIApplication


@dataclass(frozen=True)
class PolicyDefinition:
    policy_id: str
    severity: str
    rule: Dict[str, Any]
    action: str


@dataclass(frozen=True)
class PolicyViolation:
    policy_id: str
    severity: str
    ai_app: AIApplication
    reason: str


def load_policies_from_yaml(content: str) -> List[PolicyDefinition]:
    """Load policy definitions from a YAML string."""
    raw = yaml.safe_load(content) or []
    if isinstance(raw, dict) and "policies" in raw:
        raw = raw["policies"]
    if not isinstance(raw, list):
        raise ValueError("Policies YAML must define a list of policies.")
    return [_parse_policy(item) for item in raw]


def load_policies_from_file(path: str | Path) -> List[PolicyDefinition]:
    """Load policy definitions from a YAML file path."""
    content = Path(path).read_text(encoding="utf-8")
    return load_policies_from_yaml(content)


def evaluate_policies(
    ai_apps: Sequence[AIApplication],
    policies: Sequence[PolicyDefinition],
) -> List[PolicyViolation]:
    """Evaluate AI applications against policies and return violations."""
    violations: List[PolicyViolation] = []
    for app in ai_apps:
        for policy in policies:
            matched, reason = _evaluate_rule(policy.rule, app)
            if matched:
                violations.append(
                    PolicyViolation(
                        policy_id=policy.policy_id,
                        severity=policy.severity,
                        ai_app=app,
                        reason=reason,
                    )
                )
    return violations


def _parse_policy(raw: Dict[str, Any]) -> PolicyDefinition:
    if not isinstance(raw, dict):
        raise ValueError("Each policy must be a mapping.")
    for field in ("id", "severity", "rule", "action"):
        if field not in raw:
            raise ValueError(f"Policy is missing required field '{field}'.")
    return PolicyDefinition(
        policy_id=str(raw["id"]),
        severity=str(raw["severity"]),
        rule=_normalize_rule(raw["rule"]),
        action=str(raw["action"]),
    )


def _normalize_rule(rule: Any) -> Dict[str, Any]:
    if isinstance(rule, dict):
        return rule
    if isinstance(rule, list):
        return {"all": rule}
    raise ValueError("Policy rule must be a mapping or a list of conditions.")


def _evaluate_rule(rule: Dict[str, Any], app: AIApplication) -> Tuple[bool, str]:
    if "all" in rule:
        conditions = _ensure_conditions(rule["all"])
        evaluations = [_evaluate_condition(condition, app) for condition in conditions]
        matched = all(result for result, _ in evaluations)
        if matched:
            reason = _format_reason("all", evaluations)
            return True, reason
        return False, ""

    if "any" in rule:
        conditions = _ensure_conditions(rule["any"])
        evaluations = [_evaluate_condition(condition, app) for condition in conditions]
        matched = any(result for result, _ in evaluations)
        if matched:
            reason = _format_reason("any", evaluations)
            return True, reason
        return False, ""

    evaluation = _evaluate_condition(rule, app)
    if evaluation[0]:
        return True, _format_reason("single", [evaluation])
    return False, ""


def _ensure_conditions(conditions: Any) -> List[Dict[str, Any]]:
    if not isinstance(conditions, list):
        raise ValueError("Conditions must be a list.")
    normalized: List[Dict[str, Any]] = []
    for condition in conditions:
        if isinstance(condition, dict):
            normalized.append(condition)
        else:
            raise ValueError("Each condition must be a mapping.")
    return normalized


def _evaluate_condition(condition: Dict[str, Any], app: AIApplication) -> Tuple[bool, str]:
    field_path, operator, expected = _parse_condition(condition)
    values = _extract_values(app, field_path)
    if operator == "exists":
        matched = any(value is not None for value in values)
    else:
        matched = any(_compare(value, expected, operator) for value in values)
    description = _describe_condition(field_path, operator, expected, values)
    return matched, description


def _parse_condition(condition: Dict[str, Any]) -> Tuple[str, str, Any]:
    if "field" in condition:
        field_path = str(condition["field"])
        operator = str(condition.get("operator", "equals"))
        expected = condition.get("value")
        return field_path, operator, expected
    if len(condition) == 1:
        field_path, expected = next(iter(condition.items()))
        return str(field_path), "equals", expected
    raise ValueError("Condition must include 'field' or be a single-key mapping.")


def _extract_values(target: Any, path: str) -> List[Any]:
    parts = [part for part in path.split(".") if part]
    values: List[Any] = [target]

    for part in parts:
        next_values: List[Any] = []
        for value in values:
            if part == "*":
                next_values.extend(_expand_wildcard(value))
                continue
            next_values.extend(_read_attribute(value, part))
        values = next_values
        if not values:
            break

    return values


def _expand_wildcard(value: Any) -> List[Any]:
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return []


def _read_attribute(value: Any, part: str) -> List[Any]:
    if isinstance(value, dict):
        return [value.get(part)] if part in value else []
    if hasattr(value, part):
        return [getattr(value, part)]
    return []


def _compare(actual: Any, expected: Any, operator: str) -> bool:
    if operator in {"equals", "eq", "=="}:
        return actual == expected
    if operator in {"not_equals", "ne", "!="}:
        return actual != expected
    if operator == "contains":
        if isinstance(actual, (list, tuple, set)):
            return expected in actual
        if isinstance(actual, str) and isinstance(expected, str):
            return expected in actual
        return False
    if operator == "in":
        if isinstance(expected, (list, tuple, set)):
            return actual in expected
        if isinstance(expected, str) and isinstance(actual, str):
            return actual in expected
        return False
    if operator == "not_in":
        if isinstance(expected, (list, tuple, set)):
            return actual not in expected
        if isinstance(expected, str) and isinstance(actual, str):
            return actual not in expected
        return False
    if operator == "gt":
        return _numeric_compare(actual, expected, lambda a, b: a > b)
    if operator == "gte":
        return _numeric_compare(actual, expected, lambda a, b: a >= b)
    if operator == "lt":
        return _numeric_compare(actual, expected, lambda a, b: a < b)
    if operator == "lte":
        return _numeric_compare(actual, expected, lambda a, b: a <= b)
    if operator == "exists":
        return actual is not None
    raise ValueError(f"Unsupported operator '{operator}'.")


def _numeric_compare(actual: Any, expected: Any, predicate: Any) -> bool:
    if actual is None or expected is None:
        return False
    try:
        return predicate(float(actual), float(expected))
    except (TypeError, ValueError):
        return False


def _format_reason(mode: str, evaluations: Iterable[Tuple[bool, str]]) -> str:
    matched = [description for result, description in evaluations if result]
    if not matched:
        return ""
    if mode == "single":
        return f"Matched rule: {matched[0]}"
    if mode == "all":
        return "Matched all conditions: " + "; ".join(matched)
    return "Matched any condition: " + "; ".join(matched)


def _describe_condition(
    field_path: str,
    operator: str,
    expected: Any,
    values: List[Any],
) -> str:
    display_values = ", ".join(repr(value) for value in values) if values else "<none>"
    return f"{field_path} {operator} {expected!r} (found {display_values})"
