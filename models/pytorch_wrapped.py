from typing import Optional, Any
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset
from constants import CONST

from type import Evaluators, PytorchConfig
from typing import List, Tuple, Callable
from .evaluation import multiclass_classification_metrics

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
        ## Get sequence lengths
        lengths = torch.tensor([len(t) for t in features]).to(device)
        
        ## 0. Convert input features to tensors.
        features = [torch.tensor(t).type(torch.long).to(device) for t in features]
        
        ## 1. padd first element to given (constant length)
        features[0] = nn.ConstantPad1d((0, pad_length - features[0].shape[0]), 0)(features[0])
        
        ## 2. Adjust padding for all other elements
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True).T.to(device)
        labels = torch.Tensor(labels).type(torch.long).to(device).T

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
        self.model = LightningWrapper(model(self.config).to(device))
        self.evaluators: Optional[Evaluators] = evaluators
        
        self.trainer = pl.Trainer(
            accelerator=accelerator,
            max_epochs=self.config.epochs,
            # gpus=1,
            # num_nodes=8,
            # precision=16,
            # limit_train_batches=0.5,
        )

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


        self.trainer.fit(self.model, train_dataloader, val_dataloader)

    def predict(self, dataset: Dataset) -> Any:
        test_dataset = DataLoader(dataset, batch_size=32, collate_fn=hoc_collate(self.config.input_size))

        return self.trainer.predict(self.model, test_dataset)




class LightningWrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.NLLLoss()
    
    def forward(self, test_batch):
        x, y, length = test_batch

        preds,logprobs, probs = self.model(x)
        
        for name, metric in multiclass_classification_metrics:
            result = metric(y.cpu(), zip(preds.cpu().tolist(), probs.cpu().tolist()))
            print(f"{name} - {result}")
            
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y, length = train_batch
        preds,logprobs, probs=self.model(x)
        loss = self.criterion(logprobs.type(torch.float), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y, length = val_batch
        preds,logprobs, probs=self.model(x)
        loss = self.criterion(logprobs.type(torch.float), y)
        self.log("val_loss", loss)

    # def test_step(self, test_batch):
    #     x, y, length = test_batch
    #     preds, probs, hidden = self.model(x)
    
        