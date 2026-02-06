import lightning as L
from lightning.pytorch.callbacks import BatchSizeFinder, LearningRateFinder, LearningRateMonitor, EarlyStopping, ModelCheckpoint, LambdaCallback

from ..models.modules import HypergraphBetaVAE
from ..configs import DataModuleConfig, DataLoaderConfig, TrainerConfig, HuggingFaceDatasetsConfig, ModelSizeConfig, OptimizerConfig
from ..data.datamodules import HypergraphDataModule

def train_ddm(
    datamodule_config: DataModuleConfig,
    dataloader_config: DataLoaderConfig,
    trainer_config: TrainerConfig,
    huggingface_datasets_config: HuggingFaceDatasetsConfig,
    T: int,
):
    pass
