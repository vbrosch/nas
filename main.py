import collections
import random
import time
from enum import Enum
from typing import Optional, List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, AvgPool2d, MaxPool2d
from torchsummary import summary

from cifar10_loader import get_cifar10_sets

DIM = 100  # Number of bits in the bit strings (i.e. the "models").
NOISE_STDEV = 0.01  # Standard deviation of the simulated training noise.
INPUT_DIM = (3, 28, 28)
NUMBER_OF_NORMAL_CELLS_PER_STACK = 3
NUMBER_OF_BLOCKS_PER_CELL = 5

FIRST_INPUT = 0
SECOND_INPUT = 1

IN_CHANNELS = 10
OUT_CHANNELS = 10

STACK_COUNT = 3
N = 10

import torch.optim as optim

criterion = nn.CrossEntropyLoss()

NUM_EPOCHS = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader, classes = get_cifar10_sets()


########################################################################


########################################################################

class Operation(Enum):
    """
        the operation
    """
    IDENTITY = 0
    CONV_SEP_3x3 = 1
    CONV_SEP_5x5 = 2
    CONV_SEP_7x7 = 3
    DIL_CONV_SEP_3x3 = 4
    AVG_POOL_3x3 = 5
    MAX_POOL_3x3 = 6
    CONV_1x7_7x1 = 7


class MutationType(Enum):
    """
        the mutation type
    """
    CHANGE_BLOCK_INPUT = 0
    SWAP_OPERATION = 1


def _get_new_random_input_block(block_number: int, other_input_number: int):
    """
    Get a new random input block
    :return: a random input block between 0 and NUMBER_OF_BLOCKS_PER_CELL + 2 (offset for the cells input)
    """
    return random.choice(filter(lambda x: x < block_number and x != other_input_number,
                                list(random.randrange(0, NUMBER_OF_BLOCKS_PER_CELL + 2))))


def _to_conv_operation(kernel_size: int, x: any, dilation: int = 1, groups: int = 1) -> any:
    """
    Creates a conv operation
    :param kernel_size: the kernel size
    :return: the module
    """
    return F.relu(Conv2d(IN_CHANNELS, OUT_CHANNELS, kernel_size, dilation=dilation, groups=groups)(x))


def _to_operation(operation: Operation, x: any) -> any:
    """

    :param operation:
    :param x:
    :return:
    """
    if operation == Operation.IDENTITY:
        return nn.Identity()(x)
    if operation == Operation.CONV_SEP_3x3:
        return _to_conv_operation(3, x)
    if operation == Operation.CONV_SEP_5x5:
        return _to_conv_operation(3, x)
    if operation == Operation.CONV_SEP_7x7:
        return _to_conv_operation(3, x)
    if operation == Operation.DIL_CONV_SEP_3x3:
        return _to_conv_operation(3, x, dilation=2, groups=2)
    if operation == Operation.AVG_POOL_3x3:
        return AvgPool2d(3)(x)
    if operation == Operation.MAX_POOL_3x3:
        return MaxPool2d(3)(x)
    if operation == Operation.CONV_1x7_7x1:
        return F.relu(Conv2d(IN_CHANNELS, OUT_CHANNELS, (7, 1))(Conv2d(IN_CHANNELS, OUT_CHANNELS, (1, 7))(x)))


class Block(object):
    """
        Represents a block of a cell.
        A cell block combines two inputs via an operation into an output
    """

    def __init__(self, number: int, first_input_block: int, second_input_block: int, operation: Operation):
        """
        Initializes this block with the given arguments
        :param first_input_block: the first input of the block
        :param second_input_block: the second input of the block
        :param operation: the operation
        """
        self.block_number = number
        self.first_input_block = first_input_block
        self.first_input_op = self._get_random_operation()

        self.second_input_block = second_input_block
        self.second_input_op = self._get_random_operation()

        self.operation = operation

    def mutate(self) -> None:
        """
        Mutate this block by changing it's inputs or swapping it's operation
        :return: None
        """
        mutation_type = random.choice(list(MutationType))

        if mutation_type == MutationType.CHANGE_BLOCK_INPUT:
            self._mutate_input()
        else:
            self._mutate_op()

    def _mutate_input(self):
        """
        Mutate one of the inputs of the block
        :return:
        """
        inp = random.choice([FIRST_INPUT, SECOND_INPUT])

        if inp == FIRST_INPUT:
            self.first_input_block = _get_new_random_input_block(self.block_number, self.second_input_block)
        else:
            self.second_input_block = _get_new_random_input_block(self.block_number, self.first_input_block)

    @staticmethod
    def _get_random_operation() -> Operation:
        """
        get a random operation
        :return: the random operation
        """
        return random.choice(list(Operation))

    def is_block_without_dependencies(self) -> bool:
        """
        returns true if the block is only dependent of the first and second block
        :return: see above
        """
        return self.first_input_block + self.second_input_block == 1

    def _mutate_op(self):
        """
        Mutate the operation
        :return: the
        """
        selected_input = random.randint(FIRST_INPUT, SECOND_INPUT)

        if selected_input == FIRST_INPUT:
            self.first_input_op = self._get_random_operation()
        else:
            self.second_input_op = self._get_random_operation()

    def build(self, other_blocks: List[any]) -> any:
        """
        Converts this block into a pytorch module
        :return: a pytorch module
        """

        input_a = other_blocks[self.first_input_block]
        input_b = other_blocks[self.second_input_block]

        if input_a is None:
            raise RuntimeError('Input_A cannot be none')
        if input_b is None:
            raise RuntimeError('Input_B cannot be none')

        return torch.cat((_to_operation(self.first_input_op, input_a), _to_operation(self.second_input_op, input_b)))


class Cell(object):
    """
        Either a reduction or a normal cell. The cell is a building block of our neural network architecture.
        A cell is a directed acyclic graph, where each node applies an operation to an input or a previous node.

        We are using
    """

    def __init__(self):
        """
            Initializes a new instance of this cell
        """
        self.blocks: List[Block] = []

    def _get_random_block(self) -> Block:
        """
        get a block randomly
        :return: the block that was selected
        """
        return random.choice(self.blocks)

    def mutate(self):
        """
        mutate the cell
        :return:
        """
        if len(self.blocks) == 0:
            raise RuntimeError('Cannot mutate empty block list.')

        b = self._get_random_block()
        b.mutate()

    def build(self, x_0: any, x_1: any) -> any:
        """
        build a cell and convert
        :return:
        """

        if x_0 is None:
            raise RuntimeError('x_0 cannot be none')
        if x_1 is None:
            raise RuntimeError('x_1 cannot be none')

        built_blocks = [x_0, x_1]
        blocks_used_as_input = []

        for block in self.blocks:
            built_blocks.append(block.build(built_blocks))
            blocks_used_as_input += [block.first_input_block, block.second_input_block]

        # de-duplicate input list
        blocks_used_as_input = list(set(blocks_used_as_input))
        output_indices = filter(lambda x: x not in blocks_used_as_input, list(range(len(self.blocks))))
        outputs = [built_blocks[o_i] for o_i in output_indices]

        # concat all outputs
        return torch.cat(outputs)


class Model(object):
    normal_cell: Optional[Cell] = None

    reduction_cell: Optional[Cell] = None

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
      normal_cell: the normal cell architecture
      reduction_cell: the reduction cell architecture
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
        self.normal_cell = None
        self.reduction_cell = None
        self.accuracy = None

    def __str__(self):
        """Returns a readable version of both architectures."""
        normal_model = self.normal_cell.to(device)
        summary(normal_model, INPUT_DIM, device=device)

        reduction_model = self.reduction_cell.to(device)
        summary(reduction_model, INPUT_DIM, device=device)

    def _build_normal_stack(self, first_input: any, second_input: any) -> any:
        """
        build a set of normal cells
        :param first_input: the previous input
        :param second_input: the other previous input
        :return: a stack of cells
        """
        stack = []

        for n in range(N):
            stack.append(self.normal_cell.build(stack[n - 1] if n > 0 else first_input, stack[n - 2] if n > 1 else second_input))

        return stack[-1]

    def build_architecture(self, input: ) -> any:
        """
        build the full network architecture
        :return:
        """
        for stack in range(STACK_COUNT):
            normal_cell_stack = self._build_normal_stack()

    def mutate(self) -> any:
        """
        Mutates this model by mutation of either one of the cells and returns a new model with the specific mutation
        :return: the modified model
        """
        cell_to_mutate: Cell = random.choice([self.normal_cell, self.reduction_cell])
        cell_to_mutate.mutate()


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
        model.normal_cell = random_architecture()
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
        child.normal_cell = mutate_arch(parent)
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
