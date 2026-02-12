from enum import Enum, unique, auto

from hydra.core.configs import ModelSizeConfig

@unique
class ModelSize(Enum):

    @staticmethod
    def _generate_next_value_(name, start, count, last_values) -> str:
        return name.upper()

    S = auto()
    M = auto()

BVAE_CONFIGS = {
    ModelSize.S.name: ModelSizeConfig(128, 1, 2),
    ModelSize.M.name: ModelSizeConfig(256, 2, 4),
}

DDM_CONFIGS = {
    ModelSize.S.name: ModelSizeConfig(256, 2, 4),
    ModelSize.M.name: ModelSizeConfig(512, 4, 8),
}
