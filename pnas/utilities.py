import copy
import itertools
from typing import List

from modules.block import Block
from modules.cell import Cell
from search_space import Operation


def _get_vocabulary_size() -> int:
    """
    get the vocabulary size
    :return: the size of the vocabulary for the embeddings
    """
    return 0


def _get_all_possible_blocks(current_cell_size: int) -> List[Block]:
    """
    get's all possible blocks
    :param current_cell_size: the current cell sizes from (1 to ...), it is necessary, to calculate all possible inputs
    :return: all possible blocks
    """
    possible_first_input_operations = list(Operation)
    possible_second_input_operations = possible_first_input_operations.copy()
    possible_first_input = list(range(current_cell_size))
    possible_second_input = possible_first_input.copy()

    combinations = itertools.product(*[possible_first_input, possible_first_input_operations, possible_second_input,
                                       possible_second_input_operations])

    blocks = []

    for c in combinations:
        b = Block(current_cell_size)
        b.first_input_block = c[0]
        b.first_input_op = c[1]

        b.second_input_block = c[2]
        b.second_input_op = c[3]

        blocks.append(b)

    return blocks


def _expand_cell(cell: Cell) -> List[Cell]:
    """
    Expand the cell by a single block
    :param cell: the cell with block-size b
    :return: all descendant cells with block-size b + 1
    """
    expanded_cells = []
    possible_blocks = _get_all_possible_blocks(len(cell.blocks) + 2)

    for block in possible_blocks:
        tmp_cell = copy.deepcopy(cell)
        tmp_cell.blocks.append(block)
        expanded_cells.append(block)

    return expanded_cells


def _create_initial_cells() -> List[Cell]:
    """
    create all initial cells with size b = 1
    :return: the initially created cells
    """
    possible_blocks = _get_all_possible_blocks(2)
    cells = []

    for block in possible_blocks:
        c = Cell()
        c.blocks.append(block)

        cells.append(c)

    return cells


def _expand_cells(cells: List[Cell]) -> List[Cell]:
    """
    expand the cells by one more block
    :param cells: the cells that should be expanded
    :return: the expanded cells
    """
    if len(cells) == 0:
        return _create_initial_cells()

    return_cells = []

    for cell in cells:
        expanded_cells = _expand_cell(cell)
        return_cells += expanded_cells

    return return_cells