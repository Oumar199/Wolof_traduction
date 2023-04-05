"""
Runner for training the basic RNN model
-------------------------
Use AUROC as metric and log it at tensorboard 
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from custom_rnn.models.basic_rnn import BasicRNN, RNN
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAUROC
from tqdm import tqdm
import string

LETTERS = string.ascii_letters + ",.-_;"
N_LETTERS = len(LETTERS)


class BasicRnnRunner:
    def __init__(
        self,
        dataset: Dataset,
        model: nn.Module = RNN,
        version: int = 0,
        tensorboard_logdir: str = "basicrnn_logs",
    ):
        self.dataset = dataset
        self.rnn_model = model
        self.version = version
        self.logger = SummaryWriter(tensorboard_logdir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compile(
        self,
        input_size: int = N_LETTERS,
        hidden_size: int = 100,
        num_layers: int = 1,
        output_size: int = 18,
        learning_rate: float = 0.01,
        batch_size: int = 5,
        **kwargs,
    ):

        self.model = self.rnn_model(
            input_size, hidden_size, num_layers, output_size, **kwargs
        ).to(self.device)

        # self.metric = MulticlassAUROC(output_size, 'macro')

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        self.criterion = nn.NLLLoss()

        # create the dataloader
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def train(self, epochs: int = 2000, log_steps: int = 20):
        accuracy = 0
        n_steps = 0

        for epoch in tqdm(range(epochs)):
            for input_, labels, name, category in self.loader:

                input_, labels = input_.float().to(self.device), labels.long().to(
                    self.device
                ).squeeze(1)

                output = self.model(input_)

                prediction = torch.argmax(output.data, dim=1)

                loss = self.criterion(output, labels)

                accuracy += (prediction == labels).sum().item() / labels.shape[0]

                loss.backward()

                self.optimizer.step()

                self.optimizer.zero_grad()

                if (n_steps + 1) % log_steps == 0:
                    self.logger.add_scalar(
                        "Accuracy",
                        (accuracy * 100) / (log_steps),
                        global_step=n_steps + 1,
                    )
                    print(f"For last name of the batch: {name[-1]}:")
                    print(
                        f"Predicted nationality: {self.get_name_from_label(prediction[-1].item())}\
                        \nTrue nationality: {category[-1]}"
                    )
                    # auroc = 0
                    accuracy = 0
                n_steps = n_steps + 1

    def get_name_from_label(self, prediction: int):
        return self.dataset.classes[prediction]

    def predict(self, name: str):
        with torch.no_grad():
            self.model.eval()
            name = self.dataset.normalize(name)
            encoded_name = torch.zeros(len(name), N_LETTERS).to(self.device)
            for i in range(encoded_name.size(0)):
                encoded_letter = self.dataset.one_hot_encoding(name[i])
                encoded_name[i] = encoded_letter

            output = self.model(encoded_name)

            prediction = torch.argmax(output.view(1, -1).data, dim=1)

            return self.get_name_from_label(prediction.item())
