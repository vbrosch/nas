import random
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn

from modules.module_factory import _to_operation
from search_space import Operation
from utilities import _is_convolution, _is_pooling, _align_tensor, _log


def get_new_random_input_block(block_number: int):
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
        self.first_input_block = get_new_random_input_block(self.block_number)
        self.first_input_op = self.get_random_operation()
        self.first_input_module: Optional[nn.Module] = None

        self.second_input_block = get_new_random_input_block(self.block_number)
        self.second_input_op = self.get_random_operation()
        self.second_input_module: Optional[nn.Module] = None

        self.dominated_by_skip_connection = False
        self.first_input_built_via_skip_connection = False
        self.second_input_built_via_skip_connection = False
        self.output_channels = None

        self.is_normal_cell: Optional[bool] = None

    def build_ops(self, stack_num: int, stack_pos: int, is_normal_cell: bool, previous_blocks: List[any]) -> None:
        """
        build the operations
        :param is_normal_cell: whether the current cell is a normal cell
        :param stack_pos: the position of this block in the stack of cells
        :param stack_num: the stack number to which this block belongs, determines the channels of Conv2d
        :return: void
        """
        self.is_normal_cell = is_normal_cell
        self.first_input_module, first_output_channels = _to_operation(self.first_input_op, stack_num, stack_pos,
                                                                       self.first_input_block,
                                                                       is_normal_cell, previous_blocks)
        self.second_input_module, second_output_channels = _to_operation(self.second_input_op, stack_num, stack_pos,
                                                                         self.second_input_block,
                                                                         is_normal_cell, previous_blocks)

        self.dominated_by_skip_connection = self._is_dominated_by_skip_connection(previous_blocks)
        self.output_channels = first_output_channels if self._get_dominant_input() == 0 else second_output_channels

        _log("Block-{}. Output-Channel: {}".format(self.block_number, self.output_channels))

    def _is_dominated_by_skip_connection(self, previous_blocks: List[any]) -> bool:
        """
        if the output is dominated by the skip connection
        :return:
        """
        # both inputs directly connected to skip connection
        if self.first_input_block == 1 or self.first_input_block > 1 and previous_blocks[
            self.first_input_block - 2].dominated_by_skip_connection:
            self.first_input_built_via_skip_connection = True
        if self.second_input_block == 1 or self.second_input_block > 1 and previous_blocks[
            self.second_input_block - 2].dominated_by_skip_connection:
            self.second_input_built_via_skip_connection = True

        return self.first_input_built_via_skip_connection and self.second_input_built_via_skip_connection

    @staticmethod
    def get_random_operation() -> Operation:
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

    @staticmethod
    def _apply_relu_if_conv(op: Operation, t: torch.tensor) -> torch.tensor:
        """
        Applies a relu if conv else pass through
        :return:
        """
        return F.relu(t) if _is_convolution(op) else t

    @staticmethod
    def _pad_if_pooling(op: Operation, t: torch.tensor) -> torch.tensor:
        """
        Apply padding if pooling operation to remain constant sized tensor
        :param op: the operation
        :param t: the tensor
        :return: padded tensor
        """
        if _is_pooling(op):
            return F.pad(t, [1, 1, 1, 1], mode='replicate')
        return t

    def _get_dominant_input(self) -> int:
        """
        get the number of the dominant input (0, 1), A input is considered dominant, if it is not a skip connection.
        If both inputs are skip connections, we choose the input that is 'nearer' to the current block. Additionally,
        in order to gain constant filter sizes, we are boosting convolutions.
        :return: 0 if the first input is dominant, 1 if the second
        """
        # if _is_convolution(self.first_input_op) and not _is_convolution(self.second_input_op):
        #    return 0
        # elif not _is_convolution(self.first_input_op) and _is_convolution(self.second_input_op):
        #    return 1

        if not self.first_input_built_via_skip_connection and self.second_input_built_via_skip_connection:
            return 0
        elif self.first_input_built_via_skip_connection and not self.second_input_built_via_skip_connection:
            return 1

        return 0 if self.block_number - self.first_input_block <= self.block_number - self.second_input_block else 1

    def forward(self, input_a: torch.tensor, input_b: torch.tensor):
        """
        define a forward pass
        :param input_a: the first input vector/tensor
        :param input_b: the second input vector/tensor
        :return: return
        """
        _log('BLOCK-{}. IN1: {}, IN-DIM {}, OP1: {}. SKIP = {}'.format(self.block_number, self.first_input_block,
                                                                       input_a.shape,
                                                                       self.first_input_op,
                                                                       self.first_input_built_via_skip_connection))
        _log('BLOCK-{}. IN2: {}, IN-DIM {}, OP2: {}. SKIP = {}'.format(self.block_number, self.second_input_block,
                                                                       input_b.shape,
                                                                       self.second_input_op,
                                                                       self.second_input_built_via_skip_connection))

        output_a: torch.tensor = self._apply_relu_if_conv(self.first_input_op, self.first_input_module(
            self._pad_if_pooling(self.first_input_op, input_a)))
        output_b: torch.tensor = self._apply_relu_if_conv(self.second_input_op, self.second_input_module(
            self._pad_if_pooling(self.second_input_op, input_b)))

        _log('BLOCK-{}. IN1-DIM: {}, OP1: {}, OUTPUT1-DIM: {}'.format(self.block_number, self.first_input_block,
                                                                      self.first_input_op, output_a.shape))
        _log('BLOCK-{}. IN2-DIM: {}, OP2: {}, OUTPUT2-DIM: {}'.format(self.block_number, self.second_input_block,
                                                                      self.second_input_op, output_b.shape))

        # trying to set dominant input. The dominant input should be the block that is not a skip connection
        is_first_input_dominant = self._get_dominant_input() == 0

        if is_first_input_dominant:
            output_b = _align_tensor(output_b, output_a)
        else:
            output_a = _align_tensor(output_a, output_b)

        out = output_a.add(output_b)

        _log('BLOCK-{}. OUT={}. FIRST_DOM = {}'.format(self.block_number, out.shape, is_first_input_dominant))

        return out

    def __str__(self) -> str:
        """
        Convert the architecture into a string
        :return: the block
        """
        return '{} {} {} {}'.format(self.first_input_block, self.first_input_op, self.second_input_block,
                                    self.second_input_op)
