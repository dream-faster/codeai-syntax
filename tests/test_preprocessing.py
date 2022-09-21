import pandas as pd
from utils import to_token_list, write_dataframe, print_examples, read_all_files
from constants import CONST


def test_pandas_reader():
    df = read_all_files(f"{CONST.data_root_path}/original", name="training_")

    assert len(df) > 0, "DataFrame is empty"
    assert (
        df.iloc[0]["code"]
        == "def get_pyzmq_frame_buffer(frame):\n    return frame.buffer[:]\n"
    ), "The first element doesn't match"


def test_tokenizer():
    df = read_all_files(f"{CONST.data_root_path}/original", name="training_")

    df["tokenized"] = df["code"].apply(lambda x: to_token_list(x))

    assert list(df["tokenized"].iloc[0][0]) == [
        62,
        "utf-8",
        "ENCODING",
    ], "First element not correctly tokenized"
