from enum import Enum
from models.pytorch_wrapped import PytorchModel
from models.linear import Linear
from type import PytorchConfig
from utils import read_all_files
from constants import CONST
from torch.utils.data import DataLoader, Dataset

from data.python_syntax.metadata import DataParams
from data.python_syntax.dataset import CodeSyntaxDataset

import token

# Load in data
df = read_all_files(f"{CONST.data_root_path}/processed", name="training_")

num_rows = (
    df[DataParams.correct_code.value].apply(lambda x: len(x.split("\n"))).to_list()
)


config = PytorchConfig(
    hidden_size=64,
    output_size=max(num_rows),
    dictionary_size=len(token.__all__),
    val_size=0.2,
)
new_model = PytorchModel("line-predictor", config, Linear)

dataset_train = CodeSyntaxDataset(
    df, input_col=DataParams.token, label_col=DataParams.fix_location
)


new_model.fit(dataset_train)
