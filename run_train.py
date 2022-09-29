from enum import Enum
from models.pytorch_wrapped import PytorchWrapper
from models.linear import Linear
from models.rnn import Classifier
from type import PytorchWrapperConfig, PytorchModelConfig
from utils import read_all_files, load_file
from constants import CONST, ExtensionTypes
from torch.utils.data import DataLoader, Dataset

from data.python_syntax.metadata import DataParams
from data.python_syntax.dataset import CodeSyntaxDataset, CodeSyntaxPredict
from type import EmbedType, StagingConfig, dev_config, prod_config
import token
import pandas as pd


# Load in data
def train(
    staging: StagingConfig, model: PytorchWrapper, df_train: pd.DataFrame
) -> PytorchWrapper:

    dataset_train = CodeSyntaxDataset(
        df_train,
        input_col=DataParams.token_id,
        label_col=DataParams.fix_location,
    )
    model.load()

    model.fit(dataset_train)

    return model


def test(staging_config: StagingConfig, model: PytorchWrapper):
    df_test = read_all_files(
        f"{CONST.data_root_path}/processed",
        name="test_",
        limit_files=staging_config.limit_dataset,
    )

    dataset_test = CodeSyntaxDataset(
        df_test,
        input_col=DataParams.token_id,
        label_col=DataParams.fix_location,
    )
    model.test(dataset_test)


def create_model(
    staging: StagingConfig, df_train: pd.DataFrame, encoding_dict: dict
) -> PytorchWrapper:

    num_rows = df_train[DataParams.wrong_code.value].apply(lambda x: len(x)).to_list()
    num_rows_labels = (
        df_train[DataParams.metadata.value]
        .apply(lambda x: int(x[DataParams.fix_location.value]))
        .to_list()
    )

    longest_programs_tokens = (
        df_train[DataParams.token_id.value].apply(lambda x: len(x)).to_list()
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
        epochs=staging.epochs,
        batch_size=32,
        model_config=PytorchModelConfig(
            input_size=max(longest_programs_tokens),
            hidden_size=64,
            output_size=num_categories,
            dictionary_size=len(encoding_dict),
            embedding_size=120,
            embed_type=EmbedType.avarage,
        ),
    )
    new_model = PytorchWrapper("line-predictor", config, Classifier)

    return new_model


def load_dataframe(staging: StagingConfig) -> pd.DataFrame:
    df = read_all_files(
        f"{CONST.data_root_path}/processed",
        name="training_",
        limit_files=staging.limit_dataset,
    )

    return df


def get_staging_config() -> StagingConfig:
    RunningInCOLAB = (
        "google.colab" in str(get_ipython())
        if hasattr(__builtins__, "__IPYTHON__")
        else False
    )

    if RunningInCOLAB is False:
        # Try another method
        try:
            import google.colab

            RunningInCOLAB = True
        except:
            RunningInCOLAB = False

    staging_config = prod_config if RunningInCOLAB else dev_config
    return staging_config


def train_test(staging_config: StagingConfig):
    print("Running training with:")
    print(staging_config)
    df_train = load_dataframe(staging_config)
    encoding_dict = load_file(
        f"{CONST.data_root_path}/processed", name="vocab", extension=ExtensionTypes.json
    )
    model = create_model(staging_config, df_train, encoding_dict)
    trained_model = train(staging_config, model, df_train)
    test(staging_config, trained_model)


if __name__ == "__main__":
    staging_config = get_staging_config()
    train_test(staging_config)
