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


SingleTaskPredictions = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class Classifier(nn.Module):
    def __init__(self, config: PytorchModelConfig) -> None:
        super(Classifier, self).__init__()
        self.config = config
        self.encoder = RNN(config).to(device)

        self.location_predictor = nn.Linear(config.embedding_size, 1).to(device)
        self.type_predictor = nn.Linear(config.embedding_size, 3).to(device)
        self.token_predictor = nn.Linear(
            config.embedding_size, config.dictionary_size
        ).to(device)
        self.logsoftmax = nn.LogSoftmax(dim=1).to(device)
        self.dropout = nn.Dropout(0.1).to(device)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[SingleTaskPredictions, SingleTaskPredictions, SingleTaskPredictions]:
        hidden = self.encoder.initHidden(x.size(1))

        outputs = []
        for i in range(x.size(0)):
            x_hat, hidden = self.encoder(x[i], hidden)
            outputs.append(x_hat)

        encoded = torch.stack(outputs, dim=1)
        encoded = self.dropout(encoded)

        output = self.location_predictor(encoded)
        location_logprobs = self.logsoftmax(output)

        location_preds = torch.argmax(location_logprobs, dim=1)
        location_probs = torch.exp(location_logprobs)
        mod_location = (location_preds, location_logprobs, location_probs)

        selected_hidden = torch.stack(
            [data[location, :] for data, location in zip(encoded, location_preds)]
        )

        type_logprobs = self.logsoftmax(
            self.type_predictor(selected_hidden).squeeze(dim=1)
        )
        type_preds = torch.argmax(type_logprobs, dim=1)
        type_probs = torch.exp(type_logprobs)

        mod_type = (type_preds, type_logprobs, type_probs)

        token_logprobs = self.logsoftmax(
            self.token_predictor(selected_hidden).squeeze(dim=1)
        )
        token_preds = torch.argmax(type_logprobs, dim=1)
        token_probs = torch.exp(type_logprobs)

        mod_token = (token_preds, token_logprobs, token_probs)

        return mod_location, mod_type, mod_token
