from .metrics.registry import list_metrics, get_metric

def comparative_analysis(results: dict,
                         reference: dict,
                         include_metrics: list[str] | None = None,
                         exclude_metrics: list[str] | None = None,):
    
    metric_keys = include_metrics or list_metrics()

    comparison_results = dict()
    for metric_key in metric_keys:
        if exclude_metrics and metric_key in exclude_metrics:
            continue
        _Metric = get_metric(metric_key)
        metric = _Metric()
        a = results[metric_key]
        b = reference[metric_key]
        try:
            comparison_results[metric_key] = metric.compare(a, b)
        except MemoryError:
            comparison_results[metric_key] = None

    return comparison_results
