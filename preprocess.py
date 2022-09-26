import pandas as pd
from utils import to_token_list, write_dataframe, print_examples, read_all_files
from constants import CONST
from data.python_syntax.metadata import DataParams
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split


def preprocess(
    path_in: str = f"{CONST.data_root_path}/original",
    path_out: str = f"{CONST.data_root_path}/processed",
    name_in: str = "training_",
    name_out_train: str = "training_",
    name_out_test: str = "test_",
    limit_files: Optional[int] = None,
    key_to_tokenize: str = "type",
    bucket_step: int = 500,
    test_split_ratio: float = 0.2,
) -> pd.DataFrame:

    df = read_all_files(path_in, name_in, limit_files=limit_files)

    print_examples(df, 2)

    df[DataParams.token.value] = df[DataParams.code.value].apply(
        lambda x: to_token_list(x, key=key_to_tokenize)
    )
    df[DataParams.fix_location.value] = df[DataParams.metadata.value].apply(
        lambda x: x[DataParams.fix_location.value]
    )

    # df = df.sample(int(len(df) * test_split_ratio))
    split: Tuple[pd.DataFrame, pd.DataFrame] = train_test_split(
        df, test_size=test_split_ratio
    )

    train: pd.DataFrame = split[0].reset_index()
    test: pd.DataFrame = split[1].reset_index()

    for name, dataframe in [(name_out_train, train), (name_out_test, test)]:
        write_dataframe(
            dataframe,
            path_out,
            name=name,
            bucket_step=bucket_step,
        )

    return df


if __name__ == "__main__":
    preprocess(test_split_ratio=0.2)
