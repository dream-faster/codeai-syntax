class Constants:
    data_root_path: str = "data/python_syntax"
    seed: int = 73


CONST = Constants()


from enum import Enum


class TokenTypes(Enum):
    type = "type"
    string = "string"


class ExtensionTypes(Enum):
    json = "json"
