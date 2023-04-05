"""
Create a basic lstm model with pytorch
---------------------------------------------------------
We use a simple model which return hidden data and output

LSTM parameters define in https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

- input_size – The number of expected features in the input x

- hidden_size – The number of features in the hidden state h

- num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1

- bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True

- batch_first – If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details. Default: False

- dropout – If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0

- bidirectional – If True, becomes a bidirectional LSTM. Default: False

- proj_size – If > 0, will use LSTM with projections of corresponding size. Default: 0
"""

import torch
from torch import nn
from torch.nn import functional as F

activations = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
}


class BasicLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 150,
        output_size: int = 18,
        num_layers: int = 2,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
    ):

        super(BasicLSTM, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size,
        )

        self.double = 1 if not bidirectional else 2

        self.output_size = output_size

        self.classifier = nn.LogSoftmax(dim=1)

    def forward(self, input_: torch.Tensor):

        if input_.ndim == 2:
            input_ = input_.unsqueeze(0)

        outs, (hidden, cell) = self.lstm1(input_)

        out = outs[:, -1, :]

        out = out.view(out.size(0), -1)

        self.last_linear = nn.Linear(out.size(1), self.output_size).to(out.device)

        return self.classifier(out)
