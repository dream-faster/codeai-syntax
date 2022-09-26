import torch
import torch.nn as nn
from typing import Tuple
from type import PytorchModelConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    def __init__(self, config: PytorchModelConfig):
        super(RNN, self).__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.embedding = nn.Embedding(config.dictionary_size, config.hidden_size).to(
            device
        )
        self.i2h = nn.Linear(2 * config.hidden_size, config.hidden_size).to(device)
        self.i2o = nn.Linear(2 * config.hidden_size, config.hidden_size).to(device)
        self.o2o = nn.Linear(2 * config.hidden_size, config.embedding_size).to(device)

    def forward(self, input, hidden):
        input = self.embedding(input).to(device)
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)

        return output, hidden

    def initHidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size).to(device)


class Classifier(nn.Module):
    def __init__(self, config: PytorchModelConfig) -> None:
        super(Classifier, self).__init__()
        self.config = config
        self.encoder = RNN(config).to(device)

        self.predictor = nn.Linear(config.embedding_size, config.output_size).to(device)
        self.logsoftmax = nn.LogSoftmax(dim=1).to(device)
        self.dropout = nn.Dropout(0.1).to(device)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder.initHidden(x.size(1))

        outputs = []
        for i in range(x.size(0)):
            x_hat, hidden = self.encoder(x[i], hidden)
            outputs.append(x_hat)

        encoded = torch.stack(outputs, dim=2)
        # Take avarage of a source file's embedded representation.
        encoded = torch.mean(encoded, dim=2)

        encoded = self.dropout(encoded)
        output = self.predictor(encoded)
        output = self.logsoftmax(output)

        preds = torch.argmax(output, dim=1)
        logprobs = output
        probs = torch.exp(output)

        return preds, logprobs, probs
