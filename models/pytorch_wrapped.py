from typing import Optional
from dataclasses import dataclass
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from constants import CONST
from .linear import Linear
from type import Evaluators, PytorchConfig


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

    def fit(self, dataset: pd.Series, labels: Optional[pd.Series]) -> None:
        val_size = int(len(dataset) * self.config.val_size)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_dataset, batch_size=32)
        val_dataloader = DataLoader(val_dataset, batch_size=32)

        trainer = pl.Trainer(gpus=4, num_nodes=8, precision=16, limit_train_batches=0.5)
        trainer.fit(self.model, train_dataloader, val_dataloader)

    def predict(self, dataset: pd.Series) -> pd.Series:
        test_dataset = DataLoader(dataset, batch_size=32)

        return self.model(test_dataset)

    def is_fitted(self) -> bool:
        pass


class LightningWrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        embedding = self.model(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        x_hat = self.model(x)
        loss = F.mse_loss(x_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        x_hat = self.model(x)
        loss = F.mse_loss(x_hat, y)
        self.log("val_loss", loss)
