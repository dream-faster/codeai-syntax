import pandas as pd
from torch.utils.data import Dataset
from .metadata import DataParams


class CodeSyntaxDataset(Dataset):
    def __init__(
        self, data: pd.DataFrame, input_col: DataParams, label_col: DataParams
    ):
        self.input_col = input_col.value
        self.label_col = label_col.value
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = self.data[self.input_col].iloc[idx]
        label = self.data[self.label_col].iloc[idx]
        return input, label
