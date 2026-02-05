from enum import Enum, unique, auto

from ..configs import ModelSizeConfig

@unique
class ModelSize(Enum):

    @staticmethod
    def _generate_next_value_(name, start, count, last_values) -> str:
        return name.upper()

    S = auto()
    M = auto()
    L = auto()

    @property
    def cfg(self) -> ModelSizeConfig:
        return _CONFIGS[self]

_CONFIGS = {
    ModelSize.S: ModelSizeConfig(128, 1, 2),
    ModelSize.M: ModelSizeConfig(256, 2, 4),
    ModelSize.L: ModelSizeConfig(512, 3, 8),
}
