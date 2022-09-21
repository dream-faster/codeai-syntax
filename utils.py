import tokenize
from io import BytesIO
from typing import List


def to_token_list(s: str) -> List:
    tokens = []  # list of tokens extracted from source code.

    tokens = tokenize.tokenize(BytesIO(s.encode("utf-8")).readline)

    return list(tokens)


def write_tokens(tokens):
    """
    TODO: Implement your code to write extracted tokens in json format. For format of the
    JSON file kindly refer to the sample_tokens.json file.
    """

    raise Exception("Method write_tokens not implemented.")
