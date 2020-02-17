import copy
import os
from random import random
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from modules.cell import Cell
from search_space import GRAPH_OUTPUT_DIR, NUMBER_OF_NORMAL_CELLS_PER_STACK, STACK_COUNT
from utilities import _get_number_of_output_filters, _get_image_size_in_last_stack


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

        self.stack_modules = nn.ModuleList([])

        self.fc1 = nn.Linear(
            _get_number_of_output_filters() * _get_image_size_in_last_stack() * _get_image_size_in_last_stack(), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def _forward_stack(self, stack_num: int) -> [nn.Module]:
        """
        build a normal stack
        :return:
        """
        return [copy.deepcopy(self.normal_cell).build_ops(stack_num, pos, True) for pos in
                range(NUMBER_OF_NORMAL_CELLS_PER_STACK)]

    def _reduction_cell(self, stack_num: int) -> nn.Module:
        """
        get a reduction cell
        :return:
        """
        return copy.deepcopy(self.reduction_cell).build_ops(stack_num, 0, False)

    def setup_modules(self) -> None:
        """
        build modules
        :return:
        """
        for i in range(STACK_COUNT):
            self.stack_modules.append(nn.ModuleList(self._forward_stack(i)))

            if i != STACK_COUNT - 1:
                self.stack_modules.append(nn.ModuleList([self._reduction_cell(i)]))

    def forward(self, input_x) -> torch.tensor:
        """
        build the full network architecture
        :return:
        """
        print('[MODEL] Initial Shape: {}'.format(input_x.shape))

        penultimate_input: torch.tensor = input_x
        previous_input: torch.tensor = input_x

        for stack in self.stack_modules:
            print("[MODEL] -- NEW STACK --")
            stack: nn.ModuleList = stack

            penultimate_input_stack = penultimate_input
            previous_input_stack = previous_input

            out = None

            for module in stack:
                out = module(previous_input_stack, penultimate_input_stack)

                penultimate_input_stack = previous_input_stack
                previous_input_stack = out

            penultimate_input = previous_input
            previous_input = out
            print("[MODEL] -- END STACK --")

        print('BEFORE VIEW: {}'.format(previous_input.shape))
        dim = _get_number_of_output_filters() * _get_image_size_in_last_stack() * \
              _get_image_size_in_last_stack()

        print('output_filters={}'.format(_get_number_of_output_filters()))
        print('_get_image_size_in_last_stack={}'.format(_get_image_size_in_last_stack()))
        print('dim={}'.format(dim))

        x = previous_input.view(-1, dim)

        print('AFTER VIEW: {}'.format(x.shape))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        print('[MODEL] After FC: {}'.format(x.shape))

        return x

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
        normal_cell_graph = self.normal_cell.to_graph('Normal Cell')

        if not os.path.exists(GRAPH_OUTPUT_DIR):
            os.makedirs(GRAPH_OUTPUT_DIR)

        normal_cell_graph.render('normal_cell', directory=GRAPH_OUTPUT_DIR)

        reduction_cell_graph = self.reduction_cell.to_graph('Reduction Cell')
        reduction_cell_graph.render('reduction_cell', directory=GRAPH_OUTPUT_DIR)
