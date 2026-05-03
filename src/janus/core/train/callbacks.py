import logging
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

import torch
from lightning.pytorch.callbacks import Callback, EMAWeightAveraging

from janus.core.configs import DataLoaderConfig, DataModuleConfig, HuggingFaceDatasetsConfig
from janus.core.sample.ddm import sample_ddm
from janus.core.analysis.quantitative import quantitative_analysis
from janus.core.analysis.comparative import comparative_analysis
from janus.core.analysis.utils import results_discovery
from janus.core.analysis.utils import HypergraphLazyParser

logger = logging.getLogger(__name__)

class DDMSampleEvaluationCallback(Callback):

    def __init__(self,
                 datamodule_config: DataModuleConfig,
                 huggingface_datasets_config: HuggingFaceDatasetsConfig,
                 dataloader_config: DataLoaderConfig,
                 sample_every_n_steps: int = 1_000,
                 start_sampling_after_n_steps: int = 0,
                 num_samples: int = 5,
                 walk_length: int = 256,
                 references_dir: Path | str = "references",
                ):
        self.sample_every_n_steps = sample_every_n_steps
        self.start_sampling_after_n_steps = start_sampling_after_n_steps
        self.num_samples = num_samples
        self.dataloader_config = dataloader_config
        self.walk_length = walk_length
        self.datamodule_config = datamodule_config
        self.huggingface_datasets_config = huggingface_datasets_config
        self.metrics = defaultdict(list)

        for _, reference_results in results_discovery(references_dir):
            if reference_results['dataset_name'] == huggingface_datasets_config.dataset_name:
                self.reference_results = reference_results
                break
        else: # Executed after the for loop completes normally, i.e. if no break is hit
            raise ValueError(f"No reference results found for dataset {huggingface_datasets_config.dataset_name} in {references_dir}")

        assert self.sample_every_n_steps > 0, "sample_every_n_steps must be greater than 0"
        assert self.start_sampling_after_n_steps >= 0, "start_sampling_after_n_steps must be greater than or equal to 0"
        assert self.num_samples > 0, "num_samples must be greater than 0"

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Check if we should return
        global_step = trainer.global_step
        if global_step == 0 or global_step % self.sample_every_n_steps != 0:
            return

        logger = trainer.logger

        was_training = pl_module.training

        pl_module = pl_module.eval()

        ema_cb = next(cb for cb in trainer.callbacks if isinstance(cb, EMAWeightAveraging))

        ema_cb._swap_models(pl_module)

        try:
            with torch.no_grad():
                for _ in range(self.num_samples):
                    hypergraphs = sample_ddm(
                        datamodule_config=self.datamodule_config,
                        huggingface_datasets_config=self.huggingface_datasets_config,
                        dataloader_config=self.dataloader_config,
                        walk_length=self.walk_length,
                        pl_module=pl_module,
                    )
                    for hypergraph in hypergraphs:
                        hglp = HypergraphLazyParser(hypergraph)
                        results = quantitative_analysis(
                            hglp,
                            include_metrics=["hyper_net_simile", "hyper_portrait_divergence"]
                        )
                        comparison = comparative_analysis(
                            results=results,
                            reference=self.reference_results,
                            include_metrics=["hyper_net_simile", "hyper_portrait_divergence"],
                        )

                        # Store raw metrics history
                        for k, v in comparison.items():
                            self.metrics[k].append(v)

                        normalized_comparison = {}
                        normalized_values = []

                        # Normalize each metric so it is comparable with the others
                        for k, v in comparison.items():
                            values = torch.tensor(self.metrics[k], dtype=torch.float32)

                            # Baseline = first observed value
                            v0 = values[0]

                            # Use population std to avoid NaN when only one value is available
                            std = values.std(unbiased=False).item()

                            # Standardized change from baseline
                            normalized_v = (float(v) - float(v0)) / (std + 1e-8)

                            # normalized_comparison[f"{k}_normalized"] = normalized_v
                            normalized_values.append(normalized_v)

                        # Aggregate normalized metrics with the mean
                        aggregated_score = sum(abs(v) for v in normalized_values) / len(normalized_values)
                        comparison["aggregated_normalized_mean"] = aggregated_score

                        # Keep normalized metrics too
                        for k, v in normalized_comparison.items():
                            comparison[f"{k}_normalized"] = v

                        # Add comparison/ prefix to all keys
                        comparison = {f"comparison/{k}": v for k, v in comparison.items()}

                        if logger is not None:
                            logger.log_metrics(
                                comparison,
                                step=global_step,
                            )
        finally:
            ema_cb._swap_models(pl_module)

        if was_training:
            pl_module = pl_module.train()
