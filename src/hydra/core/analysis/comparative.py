# from pathlib import Path

# from datasets import load_dataset, load_from_disk
# import xgi

# from .utils import HypergraphLazyParser
# from .metrics.registry import list_metrics, get_metric

# def comparative_analysis(hg: HypergraphLazyParser,
#                  include_metrics: list[str] | None = None,
#                  split: str = "full",
#                  data_dir: Path = Path("data"),
#                  cache_dir: Path = Path("cache")):
#     dataset_name = hg.xgi_hypergraph['dataset_name']
#     dataset_dir = data_dir / dataset_name
#     local_dataset_dir = dataset_dir / "local"
#     dataset_results_path = local_dataset_dir / "hypergraph.results.json"

#     if not local_dataset_dir.exists():
#         dataset = load_dataset(dataset_name,
#                         cache_dir=str(cache_dir / "datasets"), split=split)
#         dataset.save_to_disk(local_dataset_dir)
#     else:
#         dataset = load_from_disk(local_dataset_dir)

#     if len(dataset) > 1:
#         # TODO: Not supported yet
#         raise NotImplementedError("Multiple datasets are not supported yet.")

#     reference_hypergraphs = HypergraphLazyParser(xgi.from_hif_dict(dataset[0], nodetype=int, edgetype=int))

#     metric_keys = include_metrics or list_metrics()

#     for metric_key in metric_keys:
#         _Metric = get_metric(metric_key)
#         metric = _Metric()
#         metric_value = metric.compute(hg)
