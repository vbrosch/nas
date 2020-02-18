import collections
import random
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import torch
import torch.optim as optim
from torch import nn
from torchsummary import summary

import config
from cifar10_loader import get_cifar10_sets
from config import device, NUM_EPOCHS
from modules.block import Block
from modules.cell import Cell
from modules.model import Model
from regularized_evolution.mutations import mutate_model
from search_space import NUMBER_OF_BLOCKS_PER_CELL

criterion = nn.CrossEntropyLoss()

train_loader, test_loader, classes = get_cifar10_sets()

print("Running on: {}".format(config.device))


#######################################################################


def train_and_eval(model: Model) -> float:
    """
    Train and evaluate the model

    Args:
      model: the model
    """
    config.VERBOSE = True
    model.to(device)
    print(model)
    summary(model, (3, 32, 32))

    config.VERBOSE = False

    train_network(model)
    accuracy = evaluate_architecture(model)

    return accuracy


def evaluate_architecture(net: nn.Module) -> float:
    """
    evaluate the architecture performance
    :param net: the network
    :return: accuracy
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total

    print('The network achieved the following accuracy: {}'.format(accuracy))

    return accuracy


def train_network(net: nn.Module):
    """
    Train the network on cifar 10
    :param net: the network
    :return:
    """
    optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
    start = time.perf_counter()
    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # inputs, labels = data[0], data[1]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 64 == 63:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    end = time.perf_counter()
    print('Finished Training. It took: {}'.format((end - start)))


def random_cell() -> Cell:
    """Returns a random cell with size B."""
    cell = Cell()
    for i in range(NUMBER_OF_BLOCKS_PER_CELL):
        cell.blocks.append(Block(i + 2))

    return cell


def mutate_arch(parent_arch: Model):
    """Computes the architecture for a child of the given parent architecture.

    The parent architecture is cloned and mutated to produce the child
    architecture.

    Args:
      parent_arch: the model presenting the parent

    Returns:
      An int representing the architecture (bit-string) of the child.
    """
    return mutate_model(parent_arch)


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
        model.normal_cell = random_cell()
        model.reduction_cell = random_cell()
        model.save_graphs()

        model.setup_modules()

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
        child = mutate_arch(parent)
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
        cycles=1000, population_size=10, sample_size=10)
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
