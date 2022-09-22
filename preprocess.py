import pandas as pd
from utils import to_token_list, write_dataframe, print_examples, read_all_files
from constants import CONST
from data.python_syntax.metadata import DataParams
from typing import Optional


def preprocess(
    path_in: str = f"{CONST.data_root_path}/original",
    path_out: str = f"{CONST.data_root_path}/processed",
    name: str = "training_",
    limit_files: Optional[int] = None,
    key_to_tokenize: str = "type",
    bucket_step: int = 500,
) -> pd.DataFrame:

    df = read_all_files(path_in, name, limit_files=limit_files)

    print_examples(df, 2)

    df[DataParams.token.value] = df[DataParams.code.value].apply(
        lambda x: to_token_list(x, key=key_to_tokenize)
    )
    df[DataParams.fix_location.value] = df[DataParams.metadata.value].apply(
        lambda x: x[DataParams.fix_location.value]
    )

    write_dataframe(
        df,
        path_out,
        name=name,
        bucket_step=bucket_step,
    )

    return df


if __name__ == "__main__":
    preprocess()
