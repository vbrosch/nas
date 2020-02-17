from math import floor, ceil

import torch

from config import VERBOSE
from search_space import Operation, NUMBER_OF_FILTERS, STACK_COUNT, INPUT_DIM


def _log(message: str) -> None:
    """
    log the message to stdout
    :param message: the message
    :return: void (None)
    """
    if VERBOSE:
        print(message)


def _get_image_size_in_last_stack() -> int:
    """
    get the image size in the last stack
    :return: the product of the pixels in the last stack
    """
    return int(INPUT_DIM * pow(0.5, STACK_COUNT - 1))


def _get_number_of_output_filters() -> int:
    """
    get the number of filters
    :return: the number of filters
    """
    return NUMBER_OF_FILTERS * STACK_COUNT


def _is_convolution(op: Operation) -> bool:
    """
    checks if the operation is a convolution
    :param op: the operation
    :return: true if it is a convolution operation, false if not
    """
    return op == Operation.CONV_SEP_3x3 or op == Operation.CONV_SEP_5x5 \
           or op == Operation.CONV_SEP_7x7 or op == Operation.DIL_CONV_SEP_3x3 or op == Operation.CONV_1x7_7x1


def _is_pooling(op: Operation) -> bool:
    """
    checks if the operation is pooling
    :param op: the operation
    :return: true if the operation is a pooling operation, false if not
    """
    return op == Operation.MAX_POOL_3x3 or op == Operation.AVG_POOL_3x3


def _align_tensor(a: torch.tensor, b: torch.tensor) -> torch.tensor:
    """
    align the tensors
    :param a: the tensor that should be aligned (either padded or truncated)
    :param b: the tensor that should be used as a master
    :return: the aligned tensors
    """
    if a.shape == b.shape:
        return a
    elif a.shape < b.shape:
        return _pad_tensor(a, b)
    elif a.shape > b.shape:
        return _truncate_tensor(a, b)


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


def _truncate_tensor(a: torch.tensor, b: torch.tensor) -> torch.tensor:
    """
    truncate the tensor
    :param a: the first tensor (used for modification)
    :param b: the second tensor (used for truncation)
    :return: the truncated / sliced tensor
    """
    assert a.shape > b.shape

    return a[: b.shape[0], : b.shape[1], : b.shape[2], : b.shape[3]]
