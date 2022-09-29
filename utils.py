import tokenize

# from tokenize import tokenize, generate_tokens, NUMBER, STRING, NAME, OP
from io import BytesIO
from typing import List, Optional, Any
import pandas as pd
import numpy as np
import os
import io
from constants import TokenTypes, ExtensionTypes
import json


def syntax_error_tokenizer(
    s: str, id: int, error_dict: dict, key: TokenTypes = TokenTypes.type
) -> List[tokenize.TokenInfo]:

    fp = io.StringIO(s)
    filter_types = [tokenize.ENCODING, tokenize.ENDMARKER, tokenize.ERRORTOKEN]
    tokens = []
    token_gen = tokenize.generate_tokens(fp.readline)
    while True:
        try:
            token = next(token_gen)
            if token.string and token.type not in filter_types:
                tokens.append(token._asdict()[key.value])
        except tokenize.TokenError:
            error_dict["TokenError"].append(id)
            break
        except StopIteration:
            break
        except IndentationError:
            error_dict["IndentationError"].append(id)
            continue
    return tokens


def to_token_list_correct_python(s: str, key: TokenTypes) -> List:
    tokens = []  # list of tokens extracted from source code.

    g = tokenize.tokenize(BytesIO(s.encode("utf-8")).readline)

    tokens = [t._asdict()[key.value] for t in g]

    return list(tokens)


def read_all_files(
    path: str, name: str, extension: str = "json", limit_files: Optional[int] = None
) -> pd.DataFrame:
    lst = [
        file
        for file in os.listdir(path)
        if file.split(".")[-1] == extension and name in file
    ]
    number_files = len(lst) if limit_files is None else min(limit_files, len(lst))

    df = pd.DataFrame([])
    for i in range(number_files):
        df = pd.concat([df, pd.read_json(f"{path}/{name}{i}.json")])
    return df


def write_dataframe(
    df: pd.DataFrame,
    path: str,
    name: str,
    columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None,
    bucket_step: int = 1,
) -> None:
    if os.path.exists(path) is False:
        os.makedirs(path)
    spec_columns = df.columns
    if columns is not None:
        spec_columns = columns
    if exclude_columns is not None:
        spec_columns = [
            column for column in spec_columns if column not in exclude_columns
        ]

    buckets = np.arange(0, len(df), bucket_step)
    for i in range(len(buckets)):
        df[spec_columns][buckets[i] : buckets[min(len(buckets) - 1, i + 1)]].to_json(
            f"{path}/{name}{str(i)}.json"
        )


def write_file(object: Any, path: str, name: str, extension: ExtensionTypes) -> None:
    if os.path.exists(path) is False:
        os.makedirs(path)

    with open(f"{path}/{name}.{extension.value}", "w") as f:
        write_obj = object
        if extension == ExtensionTypes.json:
            write_obj = json.dumps(object)

        f.write(write_obj)


def load_file(path: str, name: str, extension: ExtensionTypes) -> Any:
    if os.path.exists(path) is False:
        os.makedirs(path)

    with open(f"{path}/{name}.{extension.value}", "r") as f:
        load_obj = object
        if extension == ExtensionTypes.json:
            load_obj = json.load(f)

    return load_obj


def print_examples(df: pd.DataFrame, num_examples: int = 3) -> None:
    sampled = df.sample(num_examples)
    print("---")
    print(sampled["code"].iloc[0])
    print(sampled["correct_code"].iloc[0])
    print(sampled["wrong_code"].iloc[0])
    print("---")
