import torch
import torch.nn as nn
from typing import Tuple
from type import PytorchConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    def __init__(self, config: PytorchConfig):
        super(RNN, self).__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.embedding = nn.Embedding(config.dictionary_size, config.hidden_size)
        self.i2h = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.i2o = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.o2o = nn.Linear(2 * config.hidden_size, config.embedding_size)

    def forward(self, input, hidden):
        input = self.embedding(input)
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)

        return output, hidden

    def initHidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size).to(device)


class Classifier(nn.Module):
    def __init__(self, config: PytorchConfig) -> None:
        super(Classifier, self).__init__()
        self.config = config
        self.encoder = RNN(config).to(device)

        self.predictor = nn.Linear(
            config.embedding_size * config.input_size, config.output_size
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder.initHidden(x.size(1))

        encoded = torch.tensor([])
        for i in range(x.size(0)):
            x_hat, hidden = self.encoder(x[i], hidden)
            encoded = torch.cat([encoded, x_hat], dim=1)

        encoded = self.dropout(encoded)
        output = self.predictor(encoded)
        output = self.logsoftmax(output)

        preds = torch.argmax(output, dim=1)
        logprobs = output
        probs = torch.exp(output)

        return preds, logprobs, probs
