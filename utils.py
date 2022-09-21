from tokenize import tokenize, NUMBER, STRING, NAME, OP
from io import BytesIO
from typing import List, Optional
import pandas as pd
import token
import numpy as np


def to_token_list(s: str) -> List:
    tokens = []  # list of tokens extracted from source code.

    print(s)
    g = tokenize(BytesIO(s.encode("utf-8")).readline)
    for t in g:

        if t.type == NUMBER and "." in t.string:  # replace NUMBER tokens
            tokens.extend(
                [(NAME, "Decimal"), (OP, "("), (STRING, repr(t.string)), (OP, ")")]
            )
        else:
            tokens.append((t.type, t.string, token.tok_name[t.type]))

    return list(tokens)


def write_tokens(
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
