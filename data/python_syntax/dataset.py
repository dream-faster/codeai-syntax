import pandas as pd
from torch.utils.data import Dataset
from .metadata import DataParams
from typing import Tuple, List
import torch
import numpy as np


def one_hot(index: float, num_rows: int) -> pd.Series:
    ds = pd.Series(np.zeros(num_rows)).astype(float)
    ds.iloc[int(index)] = 1.0
    return ds


class CodeSyntaxDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        input_col: DataParams,
        label_col: DataParams,
        num_rows: int,
        one_hot_transform: bool = False,
    ) -> None:
        self.input = data[input_col.value].apply(
            lambda x: [float(token) for token in x]
        )

        self.label = data[label_col.value].astype(int)

        if one_hot_transform:
            self.label.apply(lambda x: one_hot(x, num_rows))

    def __len__(self):
        return len(self.input)

    def __getitem__(
        self, idx: int
    ) -> Tuple[List[float], List[float]]:  # pd.Series, pd.Series]:
        return self.input.iloc[idx], self.label.iloc[idx]
