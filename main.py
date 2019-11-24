import collections
import random
import time
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import torch
from torch import nn
from torchsummary import summary

from cifar10_loader import get_cifar10_sets

DIM = 100  # Number of bits in the bit strings (i.e. the "models").
NOISE_STDEV = 0.01  # Standard deviation of the simulated training noise.
INPUT_DIM = (3, 28, 28)
NUMBER_OF_NORMAL_CELLS_PER_STACK = 3
NUMBER_OF_BLOCKS_PER_CELL = 5

import torch.optim as optim

criterion = nn.CrossEntropyLoss()

NUM_EPOCHS = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader, classes = get_cifar10_sets()


########################################################################



########################################################################

class Model(object):
    normal_cell_arch: Optional[nn.Module]
    reduction_cell_arch: Optional[nn.Module]

    """A class representing a model.

    It holds two attributes: `arch` (the architecture) and `accuracy`
    (the accuracy / fitness). See Appendix C for an introduction to
    this toy problem.

    In the real case of neural networks, `arch` would instead hold the
    architecture of the normal and reduction cells of a neural network and
    accuracy would be instead the result of training the neural net and
    evaluating it on the validation set.

    We do not include test accuracies here as they are not used by the algorithm
    in any way. In the case of real neural networks, the test accuracy is only
    used for the purpose of reporting / plotting final results.

    In the context of evolutionary algorithms, a model is often referred to as
    an "individual".

    Attributes:
      normal_cell_arch: the normal cell architecture
      reduction_cell_arch: the reduction cell architecture
      accuracy:  the simulated validation accuracy. This is the sum of the
          bits in the bit-string, divided by DIM to produce a value in the
          interval [0.0, 1.0]. After that, a small amount of Gaussian noise is
          added with mean 0.0 and standard deviation `NOISE_STDEV`. The resulting
          number is clipped to within [0.0, 1.0] to produce the final validation
          accuracy of the model. A given model will have a fixed validation
          accuracy but two models that have the same architecture will generally
          have different validation accuracies due to this noise. In the context
          of evolutionary algorithms, this is often known as the "fitness".
    """

    def __init__(self):
        self.normal_cell_arch = None
        self.reduction_cell_arch = None
        self.accuracy = None

    def __str__(self):
        """Returns a readable version of both architectures."""
        normal_model = self.normal_cell_arch.to(device)
        summary(normal_model, INPUT_DIM, device=device)

        reduction_model = self.reduction_cell_arch.to(device)
        summary(reduction_model, INPUT_DIM, device=device)

    def build_architecture(self) -> nn.Module:
        """
        build the full network architecture
        :return:
        """
        return None


def train_and_eval(model: Model) -> float:
    """
    Train and evaluate the model

    Args:
      model: the model
    """
    net = model.build_architecture()
    train_network(net)

    accuracy = evaluate_architecture(net)

    return accuracy


def evaluate_architecture(net: nn.Module):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


def train_network(net: nn.Module):
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    start = time.perf_counter()
    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    end = time.perf_counter()
    print('Finished Training. It took: {}'.format((end - start)))


def random_architecture():
    """Returns a random architecture."""
    return random.randint(0, 2 ** DIM - 1)


def mutate_arch(parent_arch):
    """Computes the architecture for a child of the given parent architecture.

    The parent architecture is cloned and mutated to produce the child
    architecture. The child architecture is mutated by flipping a randomly chosen
    bit in its bit-string.

    Args:
      parent_arch: an int representing the architecture (bit-string) of the
          parent.

    Returns:
      An int representing the architecture (bit-string) of the child.
    """
    position = random.randint(0, DIM - 1)  # Index of the bit to flip.

    # Flip the bit at position `position` in `child_arch`.
    child_arch = parent_arch ^ (1 << position)

    return child_arch


def regularized_evolution(cycles, population_size, sample_size):
    """Algorithm for regularized evolution (i.e. aging evolution).

    Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
    Classifier Architecture Search".

    Args:
      cycles: the number of cycles the algorithm should run for.
      population_size: the number of individuals to keep in the population.
      sample_size: the number of individuals that should participate in each
          tournament.

    Returns:
      history: a list of `Model` instances, representing all the models computed
          during the evolution experiment.
    """
    population = collections.deque()
    history = []  # Not used by the algorithm, only used to report results.

    # Initialize the population with random models.
    while len(population) < population_size:
        model = Model()
        model.normal_cell_arch = random_architecture()
        model.accuracy = train_and_eval(model)
        population.append(model)
        history.append(model)

    # Carry out evolution in cycles. Each cycle produces a model and removes
    # another.
    while len(history) < cycles:
        # Sample randomly chosen models from the current population.
        sample = []
        while len(sample) < sample_size:
            # Inefficient, but written this way for clarity. In the case of neural
            # nets, the efficiency of this line is irrelevant because training neural
            # nets is the rate-determining step.
            candidate = random.choice(list(population))
            sample.append(candidate)

        # The parent is the best model in the sample.
        parent = max(sample, key=lambda i: i.accuracy)

        # Create the child model and store it.
        child = Model()
        child.normal_cell_arch = mutate_arch(parent)
        child.accuracy = train_and_eval(child)
        population.append(child)
        history.append(child)

        # Remove the oldest model.
        population.popleft()

    return history


def main() -> None:
    """
    main entrypoint
    :return: None
    """
    history = regularized_evolution(
        cycles=1000, population_size=100, sample_size=10)
    sns.set_style('white')
    x_values = range(len(history))
    y_values = [i.accuracy for i in history]
    ax = plt.gca()
    ax.scatter(
        x_values, y_values, marker='.', facecolor=(0.0, 0.0, 0.0),
        edgecolor=(0.0, 0.0, 0.0), linewidth=1, s=1)
    ax.xaxis.set_major_locator(ticker.LinearLocator(numticks=2))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_locator(ticker.LinearLocator(numticks=2))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    fig.tight_layout()
    ax.tick_params(
        axis='x', which='both', bottom='on', top='off', labelbottom='on',
        labeltop='off', labelsize=14, pad=10)
    ax.tick_params(
        axis='y', which='both', left='on', right='off', labelleft='on',
        labelright='off', labelsize=14, pad=5)
    plt.xlabel('Number of Models Evaluated', labelpad=-16, fontsize=16)
    plt.ylabel('Accuracy', labelpad=-30, fontsize=16)
    plt.xlim(0, 1000)
    sns.despine()

    plt.show()


if __name__ == '__main__':
    main()
