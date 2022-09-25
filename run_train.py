from enum import Enum
from models.pytorch_wrapped import PytorchModel
from models.linear import Linear
from models.rnn import Classifier
from type import PytorchConfig
from utils import read_all_files
from constants import CONST
from torch.utils.data import DataLoader, Dataset

from data.python_syntax.metadata import DataParams
from data.python_syntax.dataset import CodeSyntaxDataset

import token

# Load in data
def train() -> PytorchModel:
    df_train = read_all_files(
        f"{CONST.data_root_path}/processed", name="training_", limit_files=2
    )

    num_rows = (
        df_train[DataParams.correct_code.value]
        .apply(lambda x: len(x.split("\n")))
        .to_list()
    )
    num_rows_labels = (
        df_train[DataParams.metadata.value]
        .apply(lambda x: int(x[DataParams.fix_location.value]))
        .to_list()
    )

    longest_tokens = df_train[DataParams.token.value].apply(lambda x: len(x)).to_list()

    largest_row = max(num_rows + num_rows_labels) + 1
    num_categories = largest_row

    """
        Encoder: 
            num_tokens = max row length * max column length
            num_categories = max_largest_row
            in: [batch_size, num_tokens]
            out: [batch_size, num_tokens, embedding_length]
        Predictor:
            in: [batch_size, num_tokens * embedding_length] 
            out: [batch_size, num_categories]
    """

    config = PytorchConfig(
        input_size=max(longest_tokens),
        hidden_size=64,
        output_size=num_categories,
        dictionary_size=len(token.__all__),
        val_size=0.2,
        epochs=1,
        embedding_size=120,
    )
    new_model = PytorchModel("line-predictor", config, Classifier)

    dataset_train = CodeSyntaxDataset(
        df_train,
        input_col=DataParams.token,
        label_col=DataParams.fix_location,
        num_rows=largest_row,
    )

    new_model.fit(dataset_train)

    return new_model


if __name__ == "__main__":
    model = train()

    df_test = read_all_files(
        f"{CONST.data_root_path}/processed", name="test_", limit_files=1
    )

    dataset_test = CodeSyntaxDataset(
        df_test,
        input_col=DataParams.token,
        label_col=DataParams.fix_location,
    )

    model.predict(dataset_test)
