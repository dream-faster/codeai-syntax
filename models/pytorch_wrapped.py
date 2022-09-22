from typing import Optional
from dataclasses import dataclass
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from constants import CONST
from .linear import Linear
from type import Evaluators, PytorchConfig
from typing import List, Tuple, Callable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accelerator = "cuda" if torch.cuda.is_available() else "cpu"


def hoc_collate(pad_length: int) -> Callable:
    def collate_fn_padd(
        batch: List[Tuple[List[int], float]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Padds batch of variable length

        note: it converts things ToTensor manually here since the ToTensor transform
        assume it takes in images rather than arbitrary tensors.
        """
        features, labels = zip(*batch)
        ## get sequence lengths
        lengths = torch.tensor([len(t) for t in features]).to(device)

        ## padd
        features = [torch.tensor(t).type(torch.long).to(device) for t in features]
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True).T
        labels = torch.Tensor(labels).type(torch.long).to(device).T

        # features.names = ("batch", "source_tokens")
        # lengths.names = ("batch",)
        # labels.names = ("batch",)
        # mask = (features != 0).to(device)
        return features, labels, lengths

    return collate_fn_padd


class PytorchModel:
    def __init__(
        self,
        id: str,
        config: PytorchConfig,
        model: nn.Module,
        evaluators: Optional[Evaluators] = None,
    ):
        self.config = config
        self.id = id
        self.model = LightningWrapper(model(self.config))
        self.evaluators: Optional[Evaluators] = evaluators

    def load(self) -> None:
        torch.manual_seed(CONST.seed)

    def fit(self, dataset: Dataset) -> None:
        val_size = int(len(dataset) * self.config.val_size)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(
            train_dataset, batch_size=32, collate_fn=hoc_collate(self.config.input_size)
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=32, collate_fn=hoc_collate(self.config.input_size)
        )

        trainer = pl.Trainer(
            accelerator=accelerator,
            max_epochs=50,
            gpus=4,
            num_nodes=8,
            precision=16,
            limit_train_batches=0.5,
        )
        trainer.fit(self.model, train_dataloader, val_dataloader)

    def predict(self, dataset: Dataset) -> pd.Series:
        test_dataset = DataLoader(dataset, batch_size=32)

        return self.model(test_dataset)


class LightningWrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.NLLLoss()

    def forward(self, x):
        embedding = self.model(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y, length = train_batch
        hidden = self.model.initHidden(x.size(1))
        loss = 0.0
        for i in range(x.size(0)):
            x_hat, hidden = self.model(x[i], hidden)
            loss += self.criterion(x_hat.type(torch.float), y.T)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y, length = val_batch
        hidden = self.model.initHidden(x.size(1))
        loss = 0.0
        for i in range(x.size(0)):
            x_hat, hidden = self.model(x[i], hidden)
            loss += self.criterion(x_hat.type(torch.float), y.T)
        self.log("val_loss", loss)