from enum import Enum
from models.pytorch_wrapped import PytorchModel
from models.linear import Linear
from models.rnn import RNN
from type import PytorchConfig
from utils import read_all_files
from constants import CONST
from torch.utils.data import DataLoader, Dataset

from data.python_syntax.metadata import DataParams
from data.python_syntax.dataset import CodeSyntaxDataset

import token

# Load in data
df = read_all_files(
    f"{CONST.data_root_path}/processed", name="training_", limit_files=2
)

num_rows = (
    df[DataParams.correct_code.value].apply(lambda x: len(x.split("\n"))).to_list()
)
num_rows_labels = (
    df[DataParams.metadata.value]
    .apply(lambda x: int(x[DataParams.fix_location.value]))
    .to_list()
)
longest_tokens = df[DataParams.token.value].apply(lambda x: len(x)).to_list()

largest_row = max(num_rows + num_rows_labels) + 1

config = PytorchConfig(
    hidden_size=64,
    output_size=largest_row,
    dictionary_size=len(token.__all__),
    val_size=0.2,
    input_size=max(longest_tokens),
)
new_model = PytorchModel("line-predictor", config, RNN)

dataset_train = CodeSyntaxDataset(
    df,
    input_col=DataParams.token,
    label_col=DataParams.fix_location,
    num_rows=largest_row,
)


new_model.fit(dataset_train)
