from typing import List

from torch import nn
from torch.nn import AvgPool2d, MaxPool2d, Conv2d

import search_space
from search_space import IN_CHANNELS, Operation


def _get_in_channels_of_normal_cell_stack(stack_num: int) -> int:
    """
    get the input channels of the n-th stack
    :param stack_num: the stacks number [0,STACK_COUNT)
    :return: the input channels
    """
    return max(stack_num * search_space.NUMBER_OF_FILTERS, IN_CHANNELS)


def _get_output_channels_of_normal_cell_stack(stack_num: int) -> int:
    """
    get the output channels of the n-th stack
    :param stack_num: the stacks number [0,STACK_COUNT)
    :return: the output channels
    """
    return max((stack_num + 1) * search_space.NUMBER_OF_FILTERS, IN_CHANNELS)


def _get_input_channels_normal_cell(stack_num: int, stack_pos: int, input_block_num: int,
                                    previous_blocks: List[any]) -> int:
    """
    get the input channels of the convolution operation w.r.t. stack position, position inside the stack and input
    :param stack_num: determines to which stack this operation belongs
    :param stack_pos: determines the position of the operation inside the stack (only necessary for normal cells)
    :param input_block_num: the input block number of the operation
            (0 = predecessor, 1 = pre-predecessor [via skip connection])
    :return: input channels
    """
    if input_block_num == 0 or input_block_num == 1:
        if stack_pos == 0 or (stack_pos == 1 and input_block_num == 1):
            return _get_in_channels_of_normal_cell_stack(stack_num)
        else:
            return _get_output_channels_of_normal_cell_stack(stack_num)
    # if (stack_pos == 0 and (input_block_num == 0 or input_block_num == 1))
    # or (stack_pos == 1 and input_block_num == 1):
    #    return _get_in_channels_of_normal_cell_stack(stack_num)

    return previous_blocks[input_block_num - 2].output_channels


#    return _get_output_channels_of_normal_cell_stack(stack_num)


def _get_input_channels_reduction_cell(stack_num: int, input_block_num: int, previous_blocks: List[any]) -> int:
    """
    get the input channels of the convolution operation w.r.t. stack position, position inside the stack and input
    :param stack_num: determines to which stack this operation belongs
    :param input_block_num: the input block number of the operation
            (0 = predecessor, 1 = pre-predecessor [via skip connection])
    :return: input channels
    """
    # if input_block_num == 1:
    #    return _get_output_channels_of_normal_cell_stack(stack_num - 1)
    if input_block_num == 0 or input_block_num == 1:
        return _get_output_channels_of_normal_cell_stack(stack_num)

    return previous_blocks[input_block_num - 2].output_channels


def _get_input_channels(stack_num: int, stack_pos: int, input_block_num: int, previous_blocks: List[any],
                        is_normal_cell: bool) -> int:
    """
    get the input channels of the convolution operation w.r.t. stack position, position inside the stack and input
    :param stack_num: determines to which stack this operation belongs
    :param stack_pos: determines the position of the operation inside the stack (only necessary for normal cells)
    :param input_block_num: the input block number of the operation
            (0 = predecessor, 1 = pre-predecessor [via skip connection])
    :param previous_blocks: the previous blocks
    :param is_normal_cell: flag that determines if the operation belongs to a normal
            cell (true = normal cell, false = reduction cell)
    :return: input channels
    """
    if is_normal_cell:
        return _get_input_channels_normal_cell(stack_num, stack_pos, input_block_num, previous_blocks)
    return _get_input_channels_reduction_cell(stack_num, input_block_num, previous_blocks)


def _get_output_channels(stack_num: int) -> int:
    """
    get the output channels of the convolution operation w.r.t. the stack
    :param stack_num: determines to which stack this operation belongs
    :return: input channels
    """
    return _get_output_channels_of_normal_cell_stack(stack_num)


def _get_stride(input_block_num: int, is_normal_cell: bool) -> int:
    """
    Gets the stride. In normal cell, no stride is applied. In a reduction cell,
    all operations in the blocks connected to the inputs use a stride of 2 to reduce the dimensionality in half.
    :param input_block_num: the input block number
    :param is_normal_cell: flag that indicates if this cell is a normal cell. if false => reduction cell
    :return: the stride
    """
    # return 2 if not is_normal_cell and (input_block_num == 0 or input_block_num == 1) else 1
    return 1


def _get_padding(input_block_num: int, is_normal_cell: bool) -> int:
    """
    Gets the stride. In normal cell, no padding is applied. In a reduction cell,
    all operations in the blocks connected to the inputs use a padding of 1 to ensure that a kernel size 3x3 reduces
    the dimensionality in half.
    :param input_block_num: the input block number
    :param is_normal_cell: flag that indicates if this cell is a normal cell. if false => reduction cell
    :return: the stride
    """
    # return 1 if not is_normal_cell and (input_block_num == 0 or input_block_num == 1) else 0
    return 0


def _get_convolution_module(stack_num: int, stack_pos: int, input_block_num: int,
                            is_normal_cell: bool, kernel_size: int,
                            previous_blocks: List[any],
                            dilation: int = 1,
                            groups: int = 1,
                            padding: int = 0) -> nn.Module:
    """
    Creates a conv operation
    :param stack_num: the stack to which this operation belongs. determines filter size.
    :param kernel_size: the kernel size
    :return: the module
    """
    in_channels = _get_input_channels(stack_num, stack_pos, input_block_num, previous_blocks, is_normal_cell)
    out_channels = _get_output_channels(stack_num)
    stride = _get_stride(input_block_num, is_normal_cell)

    return Conv2d(in_channels, out_channels, kernel_size, dilation=dilation, groups=groups, padding=padding,
                  stride=stride,
                  padding_mode='reflection')


def _to_operation(operation: Operation, stack_num: int, stack_pos: int, input_block_num: int,
                  is_normal_cell: bool, previous_blocks: List[any]) -> (nn.Module, int):
    """
    Converts the operation label into a PyTorch module.
    :param operation: the chosen operation (from enum Operation)
    :param stack_num: determines to which stack this operation belongs
    :param stack_pos: determines the position of the operation inside the stack (only necessary for normal cells)
    :param input_block_num: the input block number of the operation
            (0 = predecessor, 1 = pre-predecessor [via skip connection])
    :param is_normal_cell: flag that determines if the operation belongs to a normal
            cell (true = normal cell, false = reduction cell)
    :return: the configured PyTorch module and the number of output channels
    """

    output_channels_eq = _get_input_channels(stack_num, stack_pos, input_block_num, previous_blocks,
                                             is_normal_cell)
    output_channels_more = _get_output_channels(stack_num)

    if operation == Operation.IDENTITY:
        return nn.Identity(), output_channels_eq
    if operation == Operation.CONV_SEP_3x3:
        return _get_convolution_module(stack_num, stack_pos, input_block_num, is_normal_cell, 3, previous_blocks,
                                       padding=1), output_channels_more
    if operation == Operation.CONV_SEP_5x5:
        return _get_convolution_module(stack_num, stack_pos, input_block_num, is_normal_cell, 5, previous_blocks,
                                       padding=2), output_channels_more
    if operation == Operation.CONV_SEP_7x7:
        return _get_convolution_module(stack_num, stack_pos, input_block_num, is_normal_cell, 7, previous_blocks,
                                       padding=3), output_channels_more
    if operation == Operation.DIL_CONV_SEP_3x3:
        return _get_convolution_module(stack_num, stack_pos, input_block_num, is_normal_cell, 3, previous_blocks,
                                       padding=2, dilation=2), output_channels_more
    if operation == Operation.AVG_POOL_3x3:
        return AvgPool2d(3, stride=_get_stride(input_block_num, is_normal_cell),
                         padding=_get_padding(input_block_num, is_normal_cell)), output_channels_eq
    if operation == Operation.MAX_POOL_3x3:
        return MaxPool2d(3, stride=_get_stride(input_block_num, is_normal_cell),
                         padding=_get_padding(input_block_num, is_normal_cell)), output_channels_eq
    if operation == Operation.CONV_1x7_7x1:
        return nn.Sequential(
            Conv2d(_get_input_channels(stack_num, stack_pos, input_block_num, previous_blocks, is_normal_cell),
                   _get_output_channels(stack_num), (7, 1), padding=(3, 0), padding_mode='reflection'),
            Conv2d(_get_output_channels(stack_num), _get_output_channels(stack_num),
                   (1, 7), padding=(0, 3), stride=_get_stride(input_block_num, is_normal_cell),
                   padding_mode='reflection')), output_channels_more
