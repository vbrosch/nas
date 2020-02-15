import random
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from modules.module_factory import _to_operation
from search_space import FIRST_INPUT, SECOND_INPUT, Operation
from search_strategy import MutationType
from utilities import _is_convolution, _pad_tensor, _is_pooling


def _get_new_random_input_block(block_number: int):
    """
    Get a new random input block
    :return: a random input block between 0 and NUMBER_OF_BLOCKS_PER_CELL + 2 (offset for the cells input)
    """
    return random.choice(range(block_number))


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

        self.is_normal_cell: Optional[bool] = None

    def build_ops(self, stack_num: int, stack_pos: int, is_normal_cell: bool) -> None:
        """
        build the operations
        :param is_normal_cell: whether the current cell is a normal cell
        :param stack_pos: the position of this block in the stack of cells
        :param stack_num: the stack number to which this block belongs, determines the channels of Conv2d
        :return: void
        """
        self.is_normal_cell = is_normal_cell
        self.first_input_module = _to_operation(self.first_input_op, stack_num, stack_pos, self.first_input_block,
                                                is_normal_cell)
        self.second_input_module = _to_operation(self.second_input_op, stack_num, stack_pos, self.second_input_block,
                                                 is_normal_cell)

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
        return F.relu(t) if _is_convolution(op) else t

    def _pad_if_pooling(self, op: Operation, input_num: int, t: torch.tensor) -> torch.tensor:
        """
        Apply padding if pooling operation to remain constant sized tensor
        :param op: the operation
        :param t: the tensor
        :return: padded tensor
        """
        if _is_pooling(op) and (self.is_normal_cell or input_num > 1):
            return F.pad(t, [t.shape[3], t.shape[3], t.shape[2], t.shape[2]], mode='replicate')
        return t

    def forward(self, input_a: torch.tensor, input_b: torch.tensor):
        """
        define a forward pass
        :param input_b: the second input vector/tensor
        :param input_a: the first input vector/tensor
        :return: return
        """
        output_a: torch.tensor = self._apply_relu_if_conv(self.first_input_op, self.first_input_module(
            self._pad_if_pooling(self.first_input_op, self.first_input_block, input_a)))
        output_b: torch.tensor = self._apply_relu_if_conv(self.second_input_op, self.second_input_module(
            self._pad_if_pooling(self.second_input_op, self.second_input_block, input_b)))

        if output_b.shape < output_a.shape:
            output_b = _pad_tensor(output_b, output_a)
        elif output_a.shape < output_b.shape:
            output_a = _pad_tensor(output_a, output_b)

        return torch.cat([output_a, output_b])
