from enum import Enum


class MutationType(Enum):
    """
        the mutation type
    """
    CHANGE_BLOCK_INPUT = 0
    SWAP_OPERATION = 1
