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

# Load in data
def predict(
    model: PytorchWrapper, data_to_predict: pd.Series
) -> Tuple[List[int], List[List[float]]]:

    predict = CodeSyntaxPredict(data_to_predict)
    predictions = model.predict(predict)
    preds, probs = predictions

    return preds, probs


def create_model() -> PytorchWrapper:
    config = PytorchWrapperConfig(val_size=0.2, epochs=1, batch_size=32)

    model = PytorchWrapper("line-predictor", config, Classifier)
    model.load("lightning_logs/version_47/checkpoints/epoch=0-step=25.ckpt")

    return model


def create_inference_dataset(strings: List[str]) -> pd.Series:
    return pd.Series(
        strings,
        dtype=str,
    ).apply(lambda x: to_token_list(x, "type"))


if __name__ == "__main__":
    model = create_model()

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
