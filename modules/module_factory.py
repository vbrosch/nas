from torch import nn
from torch.nn import AvgPool2d, MaxPool2d, Conv2d

from search_space import NUMBER_OF_FILTERS, IN_CHANNELS, Operation


def _get_in_channels_of_normal_cell_stack(stack_num: int) -> int:
    """
    get the input channels of the n-th stack
    :param stack_num: the stacks number [0,STACK_COUNT)
    :return: the input channels
    """
    return max(stack_num * NUMBER_OF_FILTERS, IN_CHANNELS)


def _get_output_channels_of_normal_cell_stack(stack_num: int) -> int:
    """
    get the output channels of the n-th stack
    :param stack_num: the stacks number [0,STACK_COUNT)
    :return: the output channels
    """
    return max((stack_num + 1) * NUMBER_OF_FILTERS, IN_CHANNELS)


def _get_input_channels_normal_cell(stack_num: int, stack_pos: int, input_block_num: int) -> int:
    """
    get the input channels of the convolution operation w.r.t. stack position, position inside the stack and input
    :param stack_num: determines to which stack this operation belongs
    :param stack_pos: determines the position of the operation inside the stack (only necessary for normal cells)
    :param input_block_num: the input block number of the operation
            (0 = predecessor, 1 = pre-predecessor [via skip connection])
    :return: input channels
    """
    if (stack_pos == 0 and (input_block_num == 0 or input_block_num == 1)) or (stack_pos == 1 and input_block_num == 1):
        return _get_in_channels_of_normal_cell_stack(stack_num)

    return _get_output_channels_of_normal_cell_stack(stack_num)


def _get_input_channels_reduction_cell(stack_num: int, input_block_num: int) -> int:
    """
    get the input channels of the convolution operation w.r.t. stack position, position inside the stack and input
    :param stack_num: determines to which stack this operation belongs
    :param input_block_num: the input block number of the operation
            (0 = predecessor, 1 = pre-predecessor [via skip connection])
    :return: input channels
    """
    if input_block_num == 1:
        return _get_output_channels_of_normal_cell_stack(stack_num - 1)

    return _get_output_channels_of_normal_cell_stack(stack_num)


def _get_input_channels(stack_num: int, stack_pos: int, input_block_num: int,
                        is_normal_cell: bool) -> int:
    """
    get the input channels of the convolution operation w.r.t. stack position, position inside the stack and input
    :param stack_num: determines to which stack this operation belongs
    :param stack_pos: determines the position of the operation inside the stack (only necessary for normal cells)
    :param input_block_num: the input block number of the operation
            (0 = predecessor, 1 = pre-predecessor [via skip connection])
    :param is_normal_cell: flag that determines if the operation belongs to a normal
            cell (true = normal cell, false = reduction cell)
    :return: input channels
    """
    if is_normal_cell:
        return _get_input_channels_normal_cell(stack_num, stack_pos, input_block_num)
    return _get_input_channels_reduction_cell(stack_num, input_block_num)


def _get_output_channels(stack_num: int) -> int:
    """
    get the output channels of the convolution operation w.r.t. the stack
    :param stack_num: determines to which stack this operation belongs
    :return: input channels
    """
    return _get_output_channels_of_normal_cell_stack(stack_num)


def _get_convolution_module(stack_num: int, stack_pos: int, input_block_num: int,
                            is_normal_cell: bool, kernel_size: int,
                            dilation: int = 1,
                            groups: int = 1) -> nn.Module:
    """
    Creates a conv operation
    :param stack_num: the stack to which this operation belongs. determines filter size.
    :param kernel_size: the kernel size
    :return: the module
    """
    in_channels = _get_input_channels(stack_num, stack_pos, input_block_num, is_normal_cell)
    out_channels = _get_output_channels(stack_num)

    return Conv2d(in_channels, out_channels, kernel_size, dilation=dilation, groups=groups)


def _to_operation(operation: Operation, stack_num: int, stack_pos: int, input_block_num: int,
                  is_normal_cell: bool) -> nn.Module:
    """
    Converts the operation label into a PyTorch module.
    :param operation: the chosen operation (from enum Operation)
    :param stack_num: determines to which stack this operation belongs
    :param stack_pos: determines the position of the operation inside the stack (only necessary for normal cells)
    :param input_block_num: the input block number of the operation
            (0 = predecessor, 1 = pre-predecessor [via skip connection])
    :param is_normal_cell: flag that determines if the operation belongs to a normal
            cell (true = normal cell, false = reduction cell)
    :return: the configured PyTorch module
    """
    if operation == Operation.IDENTITY:
        return nn.Identity()
    if operation == Operation.CONV_SEP_3x3:
        return _get_convolution_module(stack_num, stack_pos, input_block_num, is_normal_cell, 3)
    if operation == Operation.CONV_SEP_5x5:
        return _get_convolution_module(stack_num, stack_pos, input_block_num, is_normal_cell, 5)
    if operation == Operation.CONV_SEP_7x7:
        return _get_convolution_module(stack_num, stack_pos, input_block_num, is_normal_cell, 7)
    if operation == Operation.DIL_CONV_SEP_3x3:
        return _get_convolution_module(stack_num, stack_pos, input_block_num, is_normal_cell, 3, dilation=2)
    if operation == Operation.AVG_POOL_3x3:
        return AvgPool2d(3)
    if operation == Operation.MAX_POOL_3x3:
        return MaxPool2d(3)
    if operation == Operation.CONV_1x7_7x1:
        return nn.Sequential(
            Conv2d(_get_input_channels(stack_num, stack_pos, input_block_num, is_normal_cell),
                   _get_output_channels(stack_num), (7, 1)),
            Conv2d(_get_output_channels(stack_num), _get_output_channels(stack_num),
                   (1, 7)))
