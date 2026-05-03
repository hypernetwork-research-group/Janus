from __future__ import annotations

from typing import Callable, TypeVar
import re

from .protocols import Metric

TMetric = TypeVar("TMetric", bound=Metric)
TMetricCls = type[TMetric]

_REGISTRY: dict[str, TMetricCls] = {}  # (nota: per semplicità vedi sotto variante "più strict")

def register_metric(*, name: str | None = None) -> Callable[[TMetricCls], TMetricCls]:
    def _wrap(cls: TMetricCls) -> TMetricCls:
        key = name or getattr(cls, "name", re.sub(r'(?<!^)(?=[A-Z])', '_', cls.__name__).lower())
        if key in _REGISTRY:
            raise KeyError(f"Metric already registered: {key}")
        _REGISTRY[key] = cls
        return cls
    return _wrap

def get_metric(name: str) -> type[Metric]:
    # ritorno “erased” (type[Metric]) perché la registry contiene tipi eterogenei
    try:
        return _REGISTRY[name]  # type: ignore[return-value]
    except KeyError as e:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown metric '{name}'. Known: {known}") from e

def list_metrics() -> list[str]:
    return sorted(_REGISTRY.keys())
