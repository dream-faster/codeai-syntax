import pandas as pd
from torch.utils.data import Dataset
from .metadata import DataParams
from typing import Tuple, List, Optional
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
    ) -> None:
        self.input = data[input_col.value].apply(
            lambda x: [float(token) for token in x]
        )
        self.label = data[label_col.value].astype(int)

    def __len__(self):
        return len(self.input)

    def __getitem__(
        self, idx: int
    ) -> Tuple[List[float], List[float]]:  # pd.Series, pd.Series]:
        return self.input.iloc[idx], self.label.iloc[idx]


class CodeSyntaxPredict(Dataset):
    def __init__(self, data: pd.Series) -> None:
        self.input = data.apply(lambda x: [float(token) for token in x])

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx: int) -> Tuple[List[float]]:  # pd.Series, pd.Series]:
        return self.input.iloc[idx]
