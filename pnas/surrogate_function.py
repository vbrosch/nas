import torch
from torch import nn

from pnas.config import EMBEDDING_SIZE
import torch.nn.functional as F

from pnas.utilities import _get_vocabulary_size


class Surrogate(nn.Module):
    """
        the surrogate model that learns to predict the accuracies of different cells
    """

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(_get_vocabulary_size(), EMBEDDING_SIZE)
        self.fc1 = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)
        self.fc2 = nn.Linear(EMBEDDING_SIZE, 1)

    def forward(self, inp: torch.tensor) -> torch.tensor:
        """
        make a forward pass through the surrogate function
        :param inp: the input tensor
        :return: the accuracy of the network
        """
        x = self.Embedding(inp)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x
