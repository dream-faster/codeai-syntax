from typing import List, Tuple, Callable, Any, Optional
from dataclasses import dataclass

Label = int
Probabilities = List[float]

PredsWithProbs = Tuple[Label, Probabilities]


EvaluatorId = str
Evaluator = Tuple[EvaluatorId, Callable[[List, List[PredsWithProbs]], Any]]
Evaluators = List[Evaluator]

from enum import Enum


class EmbedType(Enum):
    concat = "concat"
    avarage = "avarage"


@dataclass
class PytorchModelConfig:
    input_size: int
    embedding_size: int
    hidden_size: int
    output_size: int
    dictionary_size: int
    embed_type: EmbedType


@dataclass
class PytorchWrapperConfig:
    val_size: float
    epochs: int
    batch_size: int
    model_config: Optional[PytorchModelConfig] = None


@dataclass
class HFWrapperConfig:
    val_size: float
    epochs: int
    batch_size: int


@dataclass
class StagingConfig:
    epochs: int
    limit_dataset: Optional[int] = None


dev_config = StagingConfig(limit_dataset=2, epochs=1)
prod_config = StagingConfig(limit_dataset=None, epochs=20)
