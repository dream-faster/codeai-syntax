from typing import List, Tuple, Callable, Any
from dataclasses import dataclass

Label = int
Probabilities = List[float]

PredsWithProbs = Tuple[Label, Probabilities]


EvaluatorId = str
Evaluator = Tuple[EvaluatorId, Callable[[List, List[PredsWithProbs]], Any]]
Evaluators = List[Evaluator]


@dataclass
class PytorchConfig:
    input_size: int
    embedding_size: int
    hidden_size: int
    output_size: int
    val_size: float
    dictionary_size: int
    epochs: int
