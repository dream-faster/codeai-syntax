class Constants:
    data_root_path: str = "data/python_syntax"
    model_output_path: str = "output/models"
    seed: int = 73


CONST = Constants()


from enum import Enum


class TokenTypes(Enum):
    type = "type"
    string = "string"


class ExtensionTypes(Enum):
    json = "json"


class ModelTypes(Enum):
    huggingface = "huggingace"
    pytorch_rnn = "pytorch_rnn"
