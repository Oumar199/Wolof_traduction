"""
Create a basic rnn model
-------------------------
We use a simple model which return hidden data and output
"""

import torch
from torch import nn
from torch.nn import functional as F

activations = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
}


class BasicRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(BasicRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.h_fc = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)

        self.o_fc = nn.Linear(self.input_size + self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input_, hidden):

        combination = torch.cat((input_, hidden), axis=1)

        new_hidden = F.relu(self.h_fc(combination))

        return new_hidden, self.softmax(F.relu(self.o_fc(combination)))

    def init_hidden(self):
        return torch.zeros((1, self.hidden_size))


class RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        activation: str = "tanh",
        bidirectional: bool = False,
        classify: bool = True,
        return_logits: bool = False,
    ):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.classify = classify
        self.logits = return_logits

        # we add a bidirectional attribute to find the sense of the sentence in two different direction (forward and backward). the hidden states are concatenated
        self.bidirectional = bidirectional

        # initialize the list of hidden_layers and the list of input_layers
        self.hidden_layers = nn.ModuleList()
        self.input_layers = nn.ModuleList()

        self.hidden_fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.input_fc = nn.Linear(self.input_size, self.hidden_size)

        self.hidden_layers.append(self.hidden_fc)
        self.input_layers.append(self.input_fc)

        # after the first hidden layer we create num_layers-1 layers
        if self.num_layers > 1:

            for i in range(1, num_layers):

                self.hidden_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
                self.input_layers.append(nn.Linear(self.hidden_size, self.hidden_size))

        self.duplicated = 1 if not self.bidirectional else 2

        self.o_fc = nn.Linear(self.hidden_size * self.duplicated, self.output_size)

        self.activation = activations[activation]

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_):

        # initialize the outputs
        self.outputs = []
        self.outputs_ = []
        input_ = input_.permute(1, 0, 2)
        hidden = self.init_hidden().to(input_.device)

        hiddens = (
            [hidden]
            if not self.bidirectional
            else hidden.split(self.hidden_size, dim=1)
        )

        for i in range(input_.shape[0]):

            letter = input_[i]

            hidden, output = self.forward_cell(letter, hiddens[0])

            if self.bidirectional:

                letter = input_[input_.shape[0] - i - 1]

                hidden_, output_ = self.forward_cell(letter, hiddens[1])

                self.outputs_.append(output_)

            # add a new outputs
            self.outputs.append(output)

        if self.bidirectional:

            self.outputs = torch.concatenate(
                (torch.stack(self.outputs), torch.stack(self.outputs_)), dim=2
            )

        else:

            self.outputs = torch.stack(self.outputs)

        if self.classify:

            output = self.outputs[-1, :, :].view(
                input_.size(1), self.duplicated * self.hidden_size
            )

            output = self.o_fc(output)

            if not self.return_logits:

                output = self.softmax(output)

        else:

            output = self.outputs

        return output

    def forward_cell(self, input_, hidden):

        input_out = self.input_fc(input_)

        hidden_out = self.hidden_fc(hidden[0].unsqueeze(0))

        combination = input_out + hidden_out

        new_hidden = self.activation(combination)

        # initialize the hidden states
        hidden_states = [new_hidden]

        # calculate the hidden layers' outputs
        if self.num_layers > 1:

            for i in range(1, len(self.hidden_layers)):

                input_out = self.input_layers[i](
                    new_hidden
                )  # linear layer on new hidden state at layer i-1

                hidden_out = self.hidden_layers[i](
                    hidden[i].unsqueeze(0)
                )  # linear layer on the previous hidden state at layer i

                combination = input_out + hidden_out  # combination for the layer i

                new_hidden = self.activation(
                    combination
                )  # calculate the new hidden layer for the current layer

                hidden_states.append(new_hidden)  # recuperate the hidden states

        return torch.concatenate(hidden_states, axis=0), new_hidden

    def init_hidden(self):

        return torch.zeros((self.num_layers, self.hidden_size * self.duplicated))
