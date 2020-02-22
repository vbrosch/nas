import torch
from torch import nn

from pnas.pnas_config import EMBEDDING_SIZE
import torch.nn.functional as F

from pnas.pnas_utilities import _get_vocabulary_size


class Surrogate(nn.Module):
    """
        the surrogate model that learns to predict the accuracies of different cells
    """

    def __init__(self, hidden_size: int, layers: int, dropout: float, mlp_dropout: float, mlp_layers: int,
                 mlp_hidden_size: int):
        super().__init__()

        self.layers = layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(_get_vocabulary_size(), EMBEDDING_SIZE)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, dropout=dropout)

        self.mlp_layers = mlp_layers
        self.mlp_hidden_size = mlp_hidden_size

        self.mlp = nn.Sequential()
        for i in range(self.mlp_layers):
            if i == 0:
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(self.hidden_size, self.mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
            else:
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
        self.regressor = nn.Linear(self.hidden_size if self.mlp_layers == 0 else self.mlp_hidden_size, 1)

    def forward(self, inp: torch.tensor) -> torch.tensor:
        """
        make a forward pass through the surrogate function
        :param inp: the input tensor
        :return: the accuracy of the network
        """
        x = self.Embedding(inp)
        out, _ = self.rnn(x)

        out = F.normalize(out, 2, dim=-1)

        out = torch.mean(out, dim=1)
        out = F.normalize(out, 2, dim=-1)

        out = self.mlp(out)
        out = self.regressor(out)
        predict_value = torch.sigmoid(out)

        return predict_value
