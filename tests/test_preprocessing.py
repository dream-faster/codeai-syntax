import pandas as pd
from data.python_syntax.metadata import DataParams
from utils import to_token_list, write_dataframe, print_examples, read_all_files
from constants import CONST
from preprocess import preprocess


def test_pandas_reader():
    df = read_all_files(f"{CONST.data_root_path}/original", name="training_")

    assert len(df) > 0, "DataFrame is empty"
    assert (
        df.iloc[0]["code"]
        == "def get_pyzmq_frame_buffer(frame):\n    return frame.buffer[:]\n"
    ), "The first element doesn't match"


def test_tokenizer():
    df = read_all_files(f"{CONST.data_root_path}/original", name="training_")

    df["tokenized"] = df["code"].apply(lambda x: to_token_list(x, key="type"))

    assert df["tokenized"].iloc[0][0] == 62, "First element not correctly tokenized"


def test_preprocess_pipeline():
    df = preprocess(limit_files=5)
    assert len(df) > 0, "Dataframe is empty"
    assert all(
        column in df.columns
        for column in [DataParams.fix_location.value, DataParams.token.value]
    ), "Not all preprocessing columns were created"
