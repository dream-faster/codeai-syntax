from typing import Optional, Any
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset
from constants import CONST

from type import Evaluators, PytorchWrapperConfig
from typing import List, Tuple, Callable, Union
from .evaluation import multiclass_classification_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accelerator = "cuda" if torch.cuda.is_available() else "cpu"

from type import EmbedType

def hoc_collate(pad_length: Optional[int] = None, predict:bool=False) -> Callable:
    def collate_fn_padd(
        batch: List[Tuple[List[int], float]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Padds batch of variable length

        note: it converts things ToTensor manually here since the ToTensor transform
        assume it takes in images rather than arbitrary tensors.
        """
        if predict:
            features = batch
            features = [torch.tensor(t).type(torch.long).to(device) for t in features]
            
            ## 1. padd first element to given (constant length)
            if pad_length is not None:
                features[0] = nn.ConstantPad1d((0, pad_length - features[0].shape[0]), 0)(features[0])
            features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True).T.to(device)
            
            return features
        else:
            features, labels = zip(*batch)
            ## Get sequence lengths
            lengths = torch.tensor([len(t) for t in features]).to(device)
            
            ## 0. Convert input features to tensors.
            features = [torch.tensor(t).type(torch.long).to(device) for t in features]
            
            ## 1. padd first element to given (constant length)
            if pad_length is not None:
                features[0] = nn.ConstantPad1d((0, pad_length - features[0].shape[0]), 0)(features[0])
            
            ## 2. Adjust padding for all other elements
            features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True).T.to(device)
            labels = torch.Tensor(labels).type(torch.long).to(device).T

            return features, labels, lengths

    return collate_fn_padd


class PytorchWrapper:
    def __init__(
        self,
        id: str,
        config: PytorchWrapperConfig,
        model_to_wrap: nn.Module,
        evaluators: Optional[Evaluators] = None,
    ):
        self.config = config
        self.id = id
        
        self.evaluators: Optional[Evaluators] = evaluators
        self.model_to_wrap = model_to_wrap
        self.model = None
        self.trainer = pl.Trainer(
            accelerator=accelerator,
            max_epochs=self.config.epochs,
            # gpus=1,
            # num_nodes=8,
            # precision=16,
            # limit_train_batches=0.5,
        )
        
        

    def load(self, path:Optional[str]=None) -> None:
        torch.manual_seed(CONST.seed) 
        if path:
            self.model = pl.LightningModule.load_from_checkpoint(path)
        else:
            self.pad_length = self.config.model_config.input_size if self.config.model_config.embed_type == EmbedType.concat else None
            self.model = LightningWrapper(self.model_to_wrap(self.config.model_config).to(device))
        
    def fit(self, dataset: Dataset) -> None:
        val_size = int(len(dataset) * self.config.val_size)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        
        train_dataloader = DataLoader(
            train_dataset, batch_size=32, collate_fn=hoc_collate(self.pad_length)
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=32, collate_fn=hoc_collate(self.pad_length)
        )


        self.trainer.fit(self.model, train_dataloader, val_dataloader)

    def predict(self, dataset: Dataset) -> Any:
        data_to_predict = DataLoader(dataset, batch_size=32, collate_fn=hoc_collate(self.pad_length, predict=True))
        results = self.trainer.predict(self.model, data_to_predict)
        preds, logprobs, probs = zip(*results)
        preds = torch.cat(preds)
        probs = torch.cat(probs)
        return preds, probs
    
    def test(self, dataset:Dataset):
        test_dataset = DataLoader(dataset, batch_size=32, collate_fn=hoc_collate(self.pad_length))

        self.trainer.test(self.model, test_dataset)
        results = self.model.test_results
        y, preds, logprobs, probs = zip(*results)
        y = torch.cat(y)
        preds = torch.cat(preds)
        probs = torch.cat(probs)
        for name, metric in multiclass_classification_metrics:
             result = metric(y.cpu(), zip(preds.cpu().tolist(), probs.cpu().tolist()))
             print(f"{name} - {result}")


class LightningWrapper(pl.LightningModule):
    test_results:List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.NLLLoss()
        pl.seed_everything(CONST.seed, workers=True)
        self.save_hyperparameters()
    
    def forward(self, features:torch.Tensor)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(features.size()) <2:
            features = torch.cat(features).unsqueeze(0)
        preds, logprobs, probs = self.model(features)
            
        return preds, logprobs, probs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch:torch.Tensor, batch_idx)->torch.Tensor:
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

    def test_step(self, test_batch, batch_idx):
        x, y, length = test_batch
            
        preds, probs, hidden = self.model(x)
        if not hasattr(self, 'test_results'):
            self.test_results=[(y, preds, probs, hidden)]
        else:
            self.test_results.append((y, preds, probs, hidden))
        # return preds, probs, hidden
        
        
    
        