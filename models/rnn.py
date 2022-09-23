import torch
import torch.nn as nn

from type import PytorchConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    def __init__(self, config: PytorchConfig):
        super(RNN, self).__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.embedding = nn.Embedding(config.dictionary_size, config.hidden_size)
        self.i2h = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.i2o = nn.Linear(2 * config.hidden_size, config.output_size)
        self.o2o = nn.Linear(
            config.hidden_size + config.output_size, config.output_size
        )
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input = self.embedding(input)
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size).to(device)
