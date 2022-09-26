from enum import Enum
from models.pytorch_wrapped import PytorchWrapper
from models.linear import Linear
from models.rnn import Classifier
from type import PytorchWrapperConfig, PytorchModelConfig
from utils import read_all_files
from constants import CONST
from torch.utils.data import DataLoader, Dataset

from data.python_syntax.metadata import DataParams
from data.python_syntax.dataset import CodeSyntaxDataset, CodeSyntaxPredict
from type import EmbedType
import token
import pandas as pd
from utils import to_token_list
from typing import Tuple, List
from utils import to_token_list

import os


# Load in data
def predict(
    model: PytorchWrapper, data_to_predict: pd.Series
) -> Tuple[List[int], List[List[float]]]:

    predict = CodeSyntaxPredict(data_to_predict)
    predictions = model.predict(predict)
    preds, probs = predictions

    return preds, probs


def create_model(path: str) -> PytorchWrapper:
    config = PytorchWrapperConfig(val_size=0.2, epochs=1, batch_size=32)

    model = PytorchWrapper("line-predictor", config, Classifier)
    model.load(path)

    return model


def create_inference_dataset(strings: List[str]) -> pd.Series:
    return pd.Series(
        strings,
        dtype=str,
    ).apply(lambda x: to_token_list(x, "type"))


def sort_version(e: str) -> int:
    return int(e.split("_")[-1])


def sort_epochs(e: str) -> int:
    return int(e.split("-")[0].split("=")[1])


def get_last_model_path() -> str:
    version_paths = [
        dI
        for dI in os.listdir("lightning_logs")
        if os.path.isdir(os.path.join("lightning_logs", dI))
    ]
    version_paths.sort(reverse=True, key=sort_version)

    for path in version_paths:
        if os.path.exists(f"lightning_logs/{path}/checkpoints"):
            last_version_path = path
            break

    file_names = [
        dI for dI in os.listdir(f"lightning_logs/{last_version_path}/checkpoints")
    ]
    file_names.sort(key=sort_epochs)
    last_file_name = file_names[-1]

    return f"lightning_logs/{last_version_path}/checkpoints/{last_file_name}"


def run_inference():
    path = get_last_model_path()
    model = create_model(path)

    strings_to_infer = [
        "def example(x, y):\n    return return x+y",
        "def example(x, y =):\n    return x+y",
    ]
    df_infer = create_inference_dataset(strings_to_infer)
    preds, probs = predict(model, df_infer)

    for source_code, pred, prob in zip(strings_to_infer, preds, probs):
        print(
            f"The error in the source code:\n---\n{source_code}\n---\nis at location:{pred} with probability:{prob[pred]} "
        )

    return preds, probs


if __name__ == "__main__":
    run_inference()
