import pandas as pd
from utils import (
    syntax_error_tokenizer,
    write_dataframe,
    print_examples,
    read_all_files,
    write_file,
)
from constants import CONST, TokenTypes, ExtensionTypes
from data.python_syntax.metadata import DataParams
from typing import Optional, Tuple, List
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

    df = read_all_files(path_in, name_in, limit_files=None)
    print_examples(df, 2)

    # Use tokenizer to get stem strings out of the source code
    df[DataParams.token_string.value] = df.apply(
        lambda x: syntax_error_tokenizer(
            x[DataParams.wrong_code.value],
            x[DataParams.metadata.value][DataParams.id.value],
            {"TokenError": [], "IndentationError": []},
            key=TokenTypes.string,
        ),
        axis=1,
    )

    def encode_with_vocab(vocab: dict, source_code: List[str]) -> List[int]:
        return [vocab[string] for string in source_code]

    # Turn the list of strings split from the source code to a dictionary
    vocab = set()
    for string in df[DataParams.token_string.value].to_list():
        vocab.update(string)

    vocab = {s: i for i, s in enumerate(vocab)}
    df[DataParams.token_id.value] = df[DataParams.token_string.value].apply(
        lambda x: encode_with_vocab(vocab, x)
    )

    # Make fix_location more accessible
    df[DataParams.fix_location.value] = df[DataParams.metadata.value].apply(
        lambda x: x[DataParams.fix_location.value]
    )

    # Split the data into train and test
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

    write_file(vocab, path_out, "vocab", extension=ExtensionTypes.json)

    return df


if __name__ == "__main__":
    preprocess(test_split_ratio=0.2)
