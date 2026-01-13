from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Model:
    provider: str
    model_class: str
    risk_tier: str


@dataclass
class DataSource:
    type: str
    sensitivity: str


@dataclass
class AIApplication:
    name: str
    owner: Optional[str]
    risk_level: str
    models: List[Model]
    data_sources: List[DataSource]
    capabilities: List[str]
    policies: List[str]
