import collections
import copy
import os
import random
import time
from enum import Enum
from math import floor, ceil
from typing import Optional, List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, AvgPool2d, MaxPool2d
from torchsummary import summary
from graphviz import Digraph

from cifar10_loader import get_cifar10_sets

DIM = 100  # Number of bits in the bit strings (i.e. the "models").
NOISE_STDEV = 0.01  # Standard deviation of the simulated training noise.
INPUT_DIM = (3, 28, 28)
NUMBER_OF_NORMAL_CELLS_PER_STACK = 1
NUMBER_OF_BLOCKS_PER_CELL = 5
NUMBER_OF_FILTERS = 8
GRAPH_OUTPUT_DIR = 'graphs/'

FIRST_INPUT = 0
SECOND_INPUT = 1

IN_CHANNELS = 3
OUT_CHANNELS = 3

STACK_COUNT = 1

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


def _get_new_random_input_block(block_number: int):
    """
    Get a new random input block
    :return: a random input block between 0 and NUMBER_OF_BLOCKS_PER_CELL + 2 (offset for the cells input)
    """
    return random.choice(range(block_number))


def _get_input_channels(stack_num: int, input_block_num: int, increase_filters: bool) -> int:
    """
    get the input channels according to the stack num
    :param stack_num: the stack num
    :param input_block_num: the num of the input block
    :return: input channels
    """
    if input_block_num > 1 and increase_filters:
        stack_num += 1

    return max(stack_num * NUMBER_OF_FILTERS, IN_CHANNELS)


def _get_output_channels(stack_num: int, increase_filters: bool) -> int:
    """
    get the output channels according to the stack num
    :param stack_num: the stack num
    :return: output channels
    """
    return (stack_num + 1) * NUMBER_OF_FILTERS if increase_filters else max(stack_num * NUMBER_OF_FILTERS, IN_CHANNELS)


def _to_conv_operation(stack_num: int, input_block_num: int, increase_filters: bool, kernel_size: int,
                       dilation: int = 1,
                       groups: int = 1) -> nn.Module:
    """
    Creates a conv operation
    :param stack_num: the stack to which this operation belongs. determines filter size.
    :param kernel_size: the kernel size
    :return: the module
    """
    return Conv2d(_get_input_channels(stack_num, input_block_num, increase_filters),
                  _get_output_channels(stack_num, increase_filters),
                  kernel_size,
                  dilation=dilation,
                  groups=groups)


def _is_conv_operation(op: Operation) -> bool:
    """
    is a conv operation
    :param op: the operation
    :return: true if it is a conv operation, false if not
    """
    return op == Operation.CONV_SEP_3x3 or op == Operation.CONV_SEP_5x5 or op == Operation.CONV_SEP_7x7


def _to_operation(operation: Operation, stack_num: int, input_block_num: int, increase_filters: bool) -> nn.Module:
    """

    :param operation:
    :param x:
    :return:
    """
    if operation == Operation.IDENTITY:
        return nn.Identity()
    if operation == Operation.CONV_SEP_3x3:
        return _to_conv_operation(stack_num, input_block_num, increase_filters, 3)
    if operation == Operation.CONV_SEP_5x5:
        return _to_conv_operation(stack_num, input_block_num, increase_filters, 5)
    if operation == Operation.CONV_SEP_7x7:
        return _to_conv_operation(stack_num, input_block_num, increase_filters, 7)
    if operation == Operation.DIL_CONV_SEP_3x3:
        return _to_conv_operation(stack_num, input_block_num, increase_filters, 3, dilation=2)
    if operation == Operation.AVG_POOL_3x3:
        return AvgPool2d(3)
    if operation == Operation.MAX_POOL_3x3:
        return MaxPool2d(3)
    if operation == Operation.CONV_1x7_7x1:
        return nn.Sequential(
            Conv2d(_get_input_channels(stack_num, input_block_num, increase_filters),
                   _get_output_channels(stack_num, increase_filters), (7, 1)),
            Conv2d(_get_output_channels(stack_num, increase_filters), _get_output_channels(stack_num, increase_filters),
                   (1, 7)))


def _pad_tensor(a: torch.tensor, b: torch.tensor) -> torch.tensor:
    """
    pad the tensor
    :param a: the first tensor (used for modification)
    :param b: the second tensor (used for inserting)
    :return: padded tensor with 0
    """
    assert a.shape < b.shape

    dim_difference = []

    for a_dim, b_dim in zip(a.shape, b.shape):
        if a_dim != b_dim:
            dim_difference.append(floor((b_dim - a_dim) / 2))
            dim_difference.append(ceil((b_dim - a_dim) / 2))
        else:
            dim_difference.append(0)
            dim_difference.append(0)

    dim_difference.reverse()

    c = torch.nn.functional.pad(a, dim_difference)

    return c


class Block(nn.Module):
    """
        Represents a block of a cell.
        A cell block combines two inputs via an operation into an output
    """

    def __init__(self, number: int):
        """
        Initializes this block with the given arguments
        """
        super(Block, self).__init__()

        self.block_number = number
        self.first_input_block = _get_new_random_input_block(self.block_number)
        self.first_input_op = self._get_random_operation()
        self.first_input_module: Optional[nn.Module] = None

        self.second_input_block = _get_new_random_input_block(self.block_number)
        self.second_input_op = self._get_random_operation()
        self.second_input_module: Optional[nn.Module] = None

    def build_ops(self, stack_num: int, increase_filters: bool) -> None:
        """
        build the operations
        :param stack_num: the stack number to which this cell belongs, determines the channels of Conv2d
        :return: void
        """
        self.first_input_module = _to_operation(self.first_input_op, stack_num, self.first_input_block,
                                                increase_filters)
        self.second_input_module = _to_operation(self.second_input_op, stack_num, self.second_input_block,
                                                 increase_filters)

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
            self.first_input_block = _get_new_random_input_block(self.block_number)
        else:
            self.second_input_block = _get_new_random_input_block(self.block_number)

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

    @staticmethod
    def _apply_relu_if_conv(op: Operation, t: torch.tensor) -> torch.tensor:
        """
        Applies a relu if conv else pass through
        :return:
        """
        return F.relu(t) if _is_conv_operation(op) else t

    def forward(self, input_a: torch.tensor, input_b: torch.tensor):
        """
        define a forward pass
        :param input_b: the second input vector/tensor
        :param input_a: the first input vector/tensor
        :return: return
        """
        output_a: torch.tensor = self._apply_relu_if_conv(self.first_input_op, self.first_input_module(input_a))
        output_b: torch.tensor = self._apply_relu_if_conv(self.second_input_op, self.second_input_module(input_b))

        if output_b.shape < output_a.shape:
            output_b = _pad_tensor(output_b, output_a)
        elif output_a.shape < output_b.shape:
            output_a = _pad_tensor(output_a, output_b)

        return torch.cat([output_a, output_b])


class Cell(nn.Module):
    """
        Either a reduction or a normal cell. The cell is a building block of our neural network architecture.
        A cell is a directed acyclic graph, where each node applies an operation to an input or a previous node.

        We are using
    """

    def __init__(self):
        """
            Initializes a new instance of this cell
        """
        super(Cell, self).__init__()
        self.blocks: nn.ModuleList = nn.ModuleList([])

    def _get_random_block(self) -> Block:
        """
        get a block randomly
        :return: the block that was selected
        """
        return random.choice(self.blocks)

    def build_ops(self, stack_num: int, increase_filters: bool = True) -> nn.Module:
        """
        build all block ops
        :param stack_num: the stacks number
        :return: void
        """
        for block in self.blocks:
            b: Block = block
            b.build_ops(stack_num, increase_filters)

        return self

    def mutate(self):
        """
        mutate the cell
        :return:
        """
        if len(self.blocks) == 0:
            raise RuntimeError('Cannot mutate empty block list.')

        b = self._get_random_block()
        b.mutate()

    def forward(self, input_a: torch.tensor, input_b: torch.tensor):
        """
        define a forward pass
        :param input_a: the first input vector/tensor
        :param input_b: the second input vector/tensor
        :return: a tensor of all concatenated
        """
        tensors = [input_a, input_b]
        block_used_as_input = [0, 1]

        for block in self.blocks:
            block_output = block(tensors[block.first_input_block], tensors[block.second_input_block])
            tensors.append(block_output)

            block_used_as_input += [block.first_input_block, block.second_input_block]

        output_blocks = set(range(NUMBER_OF_BLOCKS_PER_CELL + 2)) - set(block_used_as_input)

        if len(output_blocks) > 1:
            output_blocks_sorted = sorted(output_blocks, key=lambda x: tensors[x].shape, reverse=True)
            reference_tensor = tensors[output_blocks_sorted[0]]

            for t in output_blocks_sorted[1:]:
                assert reference_tensor.shape >= tensors[t].shape

                if reference_tensor.shape > tensors[t].shape:
                    tensors[t] = _pad_tensor(tensors[t], reference_tensor)

        return torch.cat([tensors[i] for i in output_blocks])

    def to_graph(self) -> Digraph:
        """
        convert this block into a digraph
        :return: the digraph
        """
        graph = Digraph()

        block_color = 'chartreuse'
        op_color = 'cadetblue1'

        graph.node('block-0', 'Input 0', color=block_color, style='filled')
        graph.node('block-1', 'Input 1', color=block_color, style='filled')

        blocks_used_as_input = []

        for b in self.blocks:
            block: Block = b

            first_input_op_node = 'block-{}-op-{}'.format(block.block_number, 1)
            second_input_op_node = 'block-{}-op-{}'.format(block.block_number, 2)

            graph.node(first_input_op_node, block.first_input_op.__str__(), color=op_color, style='filled')
            graph.node(second_input_op_node, block.second_input_op.__str__(), color=op_color, style='filled')

            graph.edge('block-{}'.format(block.first_input_block), first_input_op_node)
            graph.edge('block-{}'.format(block.second_input_block), second_input_op_node)

            graph.node('block-{}'.format(block.block_number), 'Block-{} [cat]'.format(block.block_number),
                       color=block_color, style='filled')

            graph.edge(first_input_op_node, 'block-{}'.format(block.block_number))
            graph.edge(second_input_op_node, 'block-{}'.format(block.block_number))

            blocks_used_as_input += [block.first_input_block, block.second_input_block]

        outputs = list(set(range(len(self.blocks) + 2)) - set(blocks_used_as_input))

        graph.node('output', 'Output', color=block_color, style='filled')

        for output_block in outputs:
            graph.edge('block-{}'.format(output_block), 'output')

        return graph


class Model(nn.Module):
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
        super(Model, self).__init__()

        self.normal_cell: Optional[Cell] = None
        self.reduction_cell: Optional[Cell] = None
        self.accuracy = None

        self.cell_modules = nn.ModuleList([])

        self._softmax_function = nn.Softmax(dim=1)

    def __str__(self):
        """Returns a readable version of both architectures."""
        normal_model = self.normal_cell.to(device)
        summary(normal_model, INPUT_DIM, device=device)

        reduction_model = self.reduction_cell.to(device)
        summary(reduction_model, INPUT_DIM, device=device)

    def _forward_stack(self, stack_num: int) -> [nn.Module]:
        """
        build a normal stack
        :return:
        """
        return [copy.deepcopy(self.normal_cell).build_ops(stack_num) for _ in range(NUMBER_OF_NORMAL_CELLS_PER_STACK)]

    def _reduction_cell(self, stack_num: int) -> nn.Module:
        """
        get a reduction cell
        :return:
        """
        return copy.deepcopy(self.reduction_cell).build_ops(stack_num, False)

    def setup_modules(self) -> None:
        """
        build modules
        :return:
        """
        for module in self._forward_stack(0):
            self.cell_modules.append(module)

        # self.cell_modules.append(self._reduction_cell(0))

        # for module in self._forward_stack(1):
        #    self.cell_modules.append(module)

        # self.cell_modules.append(self._reduction_cell(1))

        # for module in self._forward_stack(2):
        #    self.cell_modules.append(module)

    def forward(self, input_x):
        """
        build the full network architecture
        :return:
        """
        penultimate_input = input_x
        previous_input = input_x

        for module in self.cell_modules:
            out = module(penultimate_input, previous_input)

            penultimate_input = previous_input
            previous_input = out

        return self._softmax_function(previous_input)

    def mutate(self) -> any:
        """
        Mutates this model by mutation of either one of the cells and returns a new model with the specific mutation
        :return: the modified model
        """
        cell_to_mutate: Cell = random.choice([self.normal_cell, self.reduction_cell])
        cell_to_mutate.mutate()

    def save_graphs(self):
        """
        save a visualization
        :return:
        """
        normal_cell_graph = self.normal_cell.to_graph()
        normal_cell_graph.comment = 'Normal Cell'

        if not os.path.exists(GRAPH_OUTPUT_DIR):
            os.makedirs(GRAPH_OUTPUT_DIR)

        normal_cell_graph.render('normal_cell', directory=GRAPH_OUTPUT_DIR)


def train_and_eval(model: Model) -> float:
    """
    Train and evaluate the model

    Args:
      model: the model
    """
    summary(model, (3, 32, 32))
    model.save_graphs()

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
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


def train_network(net: nn.Module):
    """
    Train the network on cifar 10
    :param net: the network
    :return:
    """
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    start = time.perf_counter()
    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data[0].to(device), data[1].to(device)
            inputs, labels = data[0], data[1]

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
    child_arch = copy.deepcopy(parent_arch)
    child_arch.mutate()

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
        model.normal_cell = random_cell()
        model.reduction_cell = random_cell()
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
        child = Model()
        child.normal_cell = mutate_arch(parent)
        child.reduction_cell = mutate_arch(parent)
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
