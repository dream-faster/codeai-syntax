import torch
import torch.nn as nn
import torch.nn.functional as F

from type import PytorchConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Linear(nn.Module):
    def __init__(self, config: PytorchConfig) -> None:
        super(Linear, self).__init__()

        self.id = id
        self.hidden_size = config.hidden_size

        self.embedding = nn.Embedding(config.dictionary_size, config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.output_size)
        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.rename(None)
        output = self.embedding(input)
        output = F.relu(output)
        output.names = ("batch", "source_tokens", "embeddings")
        output = self.logsoftmax(self.out(output))
        return torch.argmax(output, dim=2)

    def initHidden(self) -> torch.Tensor:
        return torch.zeros(1, 1, self.hidden_size, device=device)
