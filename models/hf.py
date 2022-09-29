from typing import Optional
from transformers import (
    BertModel,
    BertConfig,
    BertTokenizer,
    TrainingArguments,
    DefaultDataCollator,
    Trainer,
)
from typing import Optional, Any
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset
from constants import CONST

from type import Evaluators, PytorchWrapperConfig, HFWrapperConfig
from typing import List, Tuple, Callable, Union
from .evaluation import multiclass_classification_metrics
from .utils import hoc_collate

# # Initializing a BERT bert-base-uncased style configuration
# configuration = BertConfig()

# # Initializing a model from the bert-base-uncased style configuration
# model = BertModel(configuration)

# # Accessing the model configuration
# configuration = model.config


# from transformers import TrainingArguments

# tokenizer = BertTokenizer(vocab)

# training_args = TrainingArguments(
#     output_dir="path/to/save/folder/",
#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=2,
# )

# from transformers import Trainer

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["test"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
# )

# trainer.train()


class HFWrapper:
    def __init__(
        self,
        id: str,
        config: HFWrapperConfig,
        vocab: dict,
        dataset_train: Dataset,
        dataset_test: Dataset,
        evaluators: Optional[Evaluators] = None,
    ):
        self.id = id
        self.config = config

        # Initializing a BERT bert-base-uncased style configuration
        configuration = BertConfig()

        # Initializing a model from the bert-base-uncased style configuration
        model = BertModel(configuration)
        tokenizer = BertTokenizer(f"{CONST.data_root_path}/processed/vocab.json")

        self.dataset_test = dataset_test

        training_args = TrainingArguments(
            output_dir=CONST.model_output_path,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=2,
        )

        self.evaluators: Optional[Evaluators] = evaluators
        self.model = None
        self.pad_length = None
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_train,
            eval_dataset=dataset_test,
            tokenizer=tokenizer,
            data_collator=hoc_collate(),  # DefaultDataCollator(),
        )

    def load(self, path: Optional[str] = None) -> None:
        torch.manual_seed(CONST.seed)

    def fit(self) -> None:
        self.trainer.train()

    def test(self) -> None:
        self.trainer.predict(test_dataset=self.dataset_test)

    # def predict(self, dataset: Dataset) -> Any:
    #     data_to_predict = DataLoader(
    #         dataset,
    #         batch_size=self.config.batch_size,
    #         collate_fn=hoc_collate(self.pad_length, predict=True),
    #     )
    #     results = self.trainer.predict(self.model, data_to_predict)
    #     preds, logprobs, probs = zip(*results)
    #     preds = torch.cat(preds)
    #     probs = torch.cat(probs)
    #     return preds, probs

    # def test(self, dataset: Dataset):
    #     test_dataset = DataLoader(
    #         dataset,
    #         batch_size=self.config.batch_size,
    #         collate_fn=hoc_collate(self.pad_length),
    #     )

    #     self.trainer.test(self.model, test_dataset)
    #     results = self.model.test_results
    #     y, preds, logprobs, probs = zip(*results)
    #     y = torch.cat(y)
    #     preds = torch.cat(preds)
    #     probs = torch.cat(probs)
    #     for name, metric in multiclass_classification_metrics:
    #         result = metric(y.cpu(), zip(preds.cpu().tolist(), probs.cpu().tolist()))
    #         print(f"{name} - {result}")
