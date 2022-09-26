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
    model_config: Optional[PytorchModelConfig] = None
