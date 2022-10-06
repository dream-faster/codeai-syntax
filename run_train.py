from enum import Enum
from models.pytorch_wrapped import PytorchWrapper
from models.linear import Linear
from models.rnn import Classifier
from models.hf import HFWrapper
from type import PytorchWrapperConfig, PytorchModelConfig, HFWrapperConfig
from utils import read_all_files, load_file
from constants import CONST, ExtensionTypes, ModelTypes
from torch.utils.data import DataLoader, Dataset

from data.python_syntax.metadata import DataParams
from data.python_syntax.dataset import CodeSyntaxDataset, CodeSyntaxPredict
from type import EmbedType, StagingConfig, dev_config, prod_config
import pandas as pd
from typing import Union, Tuple
from dataclasses import dataclass


@dataclass
class TrainDataStatistics:
    num_categories: int
    longest_programs_tokens: int


def create_model(
    model_type: ModelTypes,
    staging: StagingConfig,
    dataset_train: Dataset,
    dataset_test: Dataset,
    dataset_train_statistics: TrainDataStatistics,
    encoding_vocab: dict,
) -> PytorchWrapper:
    assert isinstance(
        model_type, ModelTypes
    ), f"model_type is not an available ModelType, Available Model types: {' ,'.join(vars(ModelTypes).keys())}"
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
    if model_type == ModelTypes.pytorch_rnn:
        config = PytorchWrapperConfig(
            val_size=0.2,
            epochs=staging.epochs,
            batch_size=32,
            model_config=PytorchModelConfig(
                input_size=dataset_train_statistics.longest_programs_tokens,
                hidden_size=64,
                output_size=dataset_train_statistics.num_categories,
                dictionary_size=len(encoding_vocab),
                embedding_size=120,
                embed_type=EmbedType.avarage,
            ),
        )
        new_model = PytorchWrapper(
            "line-predictor", config, Classifier, dataset_train, dataset_test
        )
    elif model_type == ModelTypes.huggingface:

        config = HFWrapperConfig(val_size=0.2, epochs=staging.epochs, batch_size=32)
        new_model = HFWrapper(
            "line-predictor",
            config,
            encoding_vocab,
            dataset_train,
            dataset_test,
        )

    elif model_type == ModelTypes.lstm:
        new_model = HFWrapper(
            "line-predictor",
            config,
            encoding_vocab,
            dataset_train,
            dataset_test,
        )

    return new_model


def load_dataframe(name: str, staging: StagingConfig) -> pd.DataFrame:
    df = read_all_files(
        f"{CONST.data_root_path}/processed",
        name=name,
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


def get_datasets(
    staging_config: StagingConfig,
) -> Tuple[Dataset, Dataset, TrainDataStatistics]:
    df_train = load_dataframe("training_", staging_config)
    df_test = load_dataframe("test_", staging_config)
    dataset_train = CodeSyntaxDataset(
        df_train,
        input_col=DataParams.token_id,
        label_col=DataParams.fix_location,
    )
    dataset_test = CodeSyntaxDataset(
        df_test,
        input_col=DataParams.token_id,
        label_col=DataParams.fix_location,
    )

    num_rows = df_train[DataParams.wrong_code.value].apply(lambda x: len(x)).to_list()
    num_rows_labels = (
        df_train[DataParams.metadata.value]
        .apply(lambda x: int(x[DataParams.fix_location.value]))
        .to_list()
    )

    longest_programs_tokens = max(
        df_train[DataParams.token_id.value].apply(lambda x: len(x)).to_list()
    )

    num_categories = max(num_rows + num_rows_labels) + 1

    dataset_train_statistics = TrainDataStatistics(
        num_categories=num_categories, longest_programs_tokens=longest_programs_tokens
    )

    return dataset_train, dataset_test, dataset_train_statistics


def train_test(staging_config: StagingConfig):
    print("Running training with:")
    print(staging_config)

    dataset_train, dataset_test, dataset_train_statistics = get_datasets(staging_config)
    encoding_vocab = load_file(
        f"{CONST.data_root_path}/processed", name="vocab", extension=ExtensionTypes.json
    )

    model = create_model(
        ModelTypes.huggingface,
        staging_config,
        dataset_train,
        dataset_test,
        dataset_train_statistics,
        encoding_vocab,
    )

    model.load()
    model.fit()
    model.test()


if __name__ == "__main__":
    staging_config = get_staging_config()
    train_test(staging_config)
