from tokenize import tokenize, NUMBER, STRING, NAME, OP
from io import BytesIO
from typing import List, Optional
import pandas as pd
import numpy as np
import os


def to_token_list(s: str, key: str) -> List:
    tokens = []  # list of tokens extracted from source code.

    g = tokenize(BytesIO(s.encode("utf-8")).readline)
    for t in g:

        # if t.type == NUMBER and "." in t.string:  # replace NUMBER tokens
        #     tokens.extend(
        #         [(NAME, "Decimal"), (OP, "("), (STRING, repr(t.string)), (OP, ")")]
        #     )
        # else:
        tokens.append(t._asdict()[key])  # (t.type, t.string, token.tok_name[t.type]))

    return list(tokens)


def read_all_files(
    path: str, name: str, extension: str = "json", limit_files: Optional[int] = None
) -> pd.DataFrame:
    lst = [file for file in os.listdir(path) if file.split(".")[-1] == extension]
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


def print_examples(df: pd.DataFrame, num_examples: int = 3) -> None:
    sampled = df.sample(num_examples)
    print("---")
    print(sampled["code"].iloc[0])
    print(sampled["correct_code"].iloc[0])
    print(sampled["wrong_code"].iloc[0])
    print("---")
