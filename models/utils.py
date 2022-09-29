from typing import Optional, List, Tuple, Callable, Union
import torch

from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hoc_collate(pad_length: Optional[int] = None, predict: bool = False) -> Callable:
    def collate_fn_padd(
        batch: List[Tuple[List[int], float]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Padds batch of variable length
        """
        if predict:
            features = batch
            features = [torch.tensor(t).type(torch.long).to(device) for t in features]

            ## 1. padd first element to given (constant length)
            if pad_length is not None:
                features[0] = nn.ConstantPad1d(
                    (0, pad_length - features[0].shape[0]), 0
                )(features[0])
            features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True).T.to(
                device
            )

            return features
        else:
            features, labels = zip(*batch)
            ## Get sequence lengths
            lengths = torch.tensor([len(t) for t in features]).to(device)

            ## 0. Convert input features to tensors.
            features = [torch.tensor(t).type(torch.long).to(device) for t in features]

            ## 1. padd first element to given (constant length)
            if pad_length is not None:
                features[0] = nn.ConstantPad1d(
                    (0, pad_length - features[0].shape[0]), 0
                )(features[0])

            ## 2. Adjust padding for all other elements
            features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True).T.to(
                device
            )
            labels = torch.Tensor(labels).type(torch.long).to(device).T

            return features, labels, lengths

    return collate_fn_padd
