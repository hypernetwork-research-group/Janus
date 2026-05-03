from typing import Protocol, TypeAlias, runtime_checkable, ClassVar

from janus.core.analysis.utils import HypergraphLazyParser

type MetricResult = int | float | list | dict

@runtime_checkable
class Metric(Protocol):

    name: ClassVar[str]
    description: ClassVar[str]

    def compute(self, hg: HypergraphLazyParser) -> MetricResult: ...

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult: ...

MetricType: TypeAlias = type[Metric]
