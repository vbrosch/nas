from enum import Enum

NUMBER_OF_NORMAL_CELLS_PER_STACK = 1
NUMBER_OF_BLOCKS_PER_CELL = 3
NUMBER_OF_FILTERS = 4
GRAPH_OUTPUT_DIR = 'graphs/'

FIRST_INPUT = 0
SECOND_INPUT = 1

IN_CHANNELS = 3
OUT_CHANNELS = 3

STACK_COUNT = 2

INPUT_DIM = 32


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
