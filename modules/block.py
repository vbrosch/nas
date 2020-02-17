import random
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn

from modules.module_factory import _to_operation
from search_space import FIRST_INPUT, SECOND_INPUT, Operation
from search_strategy import MutationType
from utilities import _is_convolution, _pad_tensor, _is_pooling, _align_tensor, _log


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

        self.output_channels = first_output_channels if self._get_dominant_input() == 0 else second_output_channels

        _log("Block-{}. Output-Channel: {}".format(self.block_number, self.output_channels))

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

        # TODO: Comment
        if self.first_input_block == 0 and self.second_input_block < 2:
            return 0
        if self.second_input_block == 0 and self.first_input_block < 2:
            return 1

        return 0 if self.block_number - self.first_input_block <= self.block_number - self.second_input_block else 1

    def forward(self, input_a: torch.tensor, input_b: torch.tensor):
        """
        define a forward pass
        :param input_a: the first input vector/tensor
        :param input_b: the second input vector/tensor
        :return: return
        """
        _log('BLOCK-{}. IN1: {}, IN-DIM {}, OP1: {}'.format(self.block_number, self.first_input_block, input_a.shape,
                                                            self.first_input_op))
        _log('BLOCK-{}. IN2: {}, IN-DIM {}, OP2: {}'.format(self.block_number, self.second_input_block, input_b.shape,
                                                            self.second_input_op))

        output_a: torch.tensor = self._apply_relu_if_conv(self.first_input_op, self.first_input_module(
            self._pad_if_pooling(self.first_input_op, self.first_input_block, input_a)))
        output_b: torch.tensor = self._apply_relu_if_conv(self.second_input_op, self.second_input_module(
            self._pad_if_pooling(self.second_input_op, self.second_input_block, input_b)))

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

        _log('BLOCK-{}. OUT={}'.format(self.block_number, out.shape))

        return out
