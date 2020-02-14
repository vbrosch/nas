from math import floor, ceil

import torch

from search_space import Operation


def _is_convolution(op: Operation) -> bool:
    """
    checks if the operation is a convolution
    :param op: the operation
    :return: true if it is a convolution operation, false if not
    """
    return op == Operation.CONV_SEP_3x3 or op == Operation.CONV_SEP_5x5 or op == Operation.CONV_SEP_7x7


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
