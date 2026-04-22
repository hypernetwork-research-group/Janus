import logging

logger = logging.getLogger(__name__)

from tqdm import tqdm

from .utils import HypergraphLazyParser
from .metrics.registry import list_metrics, get_metric

def quantitative_analysis(hg: HypergraphLazyParser,
                 include_metrics: list[str] | None = None,
                 exclude_metrics: list[str] | None = None) -> dict[str, float | int | list | dict]:

    metric_keys = include_metrics or list_metrics()
    print(hg)

    results = dict()
    for metric_key in tqdm(metric_keys, desc=str(hg), leave=False):
        if exclude_metrics and metric_key in exclude_metrics and metric_key not in include_metrics:
            continue
        _Metric = get_metric(metric_key)
        metric = _Metric()
        print(metric)
        try:
            metric_value = metric.compute(hg)
        except MemoryError:
            logger.warning(f"MemoryError computing metric {metric_key} for hypergraph {hg}")
            metric_value = None
        results[metric_key] = metric_value
    return results
