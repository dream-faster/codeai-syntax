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


# Load in data
def train(model: PytorchWrapper, df_train: pd.DataFrame) -> PytorchWrapper:

    dataset_train = CodeSyntaxDataset(
        df_train,
        input_col=DataParams.token,
        label_col=DataParams.fix_location,
    )
    model.load()

    model.fit(dataset_train)

    return model


def test(model: PytorchWrapper):
    df_test = read_all_files(
        f"{CONST.data_root_path}/processed", name="test_", limit_files=1
    )

    dataset_test = CodeSyntaxDataset(
        df_test,
        input_col=DataParams.token,
        label_col=DataParams.fix_location,
    )
    model.test(dataset_test)


def create_model(df_train: pd.DataFrame) -> PytorchWrapper:

    num_rows = df_train[DataParams.correct_code.value].apply(lambda x: len(x)).to_list()
    num_rows_labels = (
        df_train[DataParams.metadata.value]
        .apply(lambda x: int(x[DataParams.fix_location.value]))
        .to_list()
    )

    longest_programs_tokens = (
        df_train[DataParams.token.value].apply(lambda x: len(x)).to_list()
    )

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

    config = PytorchWrapperConfig(
        val_size=0.2,
        epochs=1,
        model_config=PytorchModelConfig(
            input_size=max(longest_programs_tokens),
            hidden_size=64,
            output_size=num_categories,
            dictionary_size=len(token.__all__),
            embedding_size=120,
            embed_type=EmbedType.avarage,
        ),
    )
    new_model = PytorchWrapper("line-predictor", config, Classifier)

    return new_model


def load_dataframe() -> pd.DataFrame:
    df = read_all_files(
        f"{CONST.data_root_path}/processed", name="training_", limit_files=2
    )

    return df


def train_test():
    df_train = load_dataframe()
    model = create_model(df_train)
    trained_model = train(model, df_train)
    test(trained_model)


if __name__ == "__main__":
    train_test()
