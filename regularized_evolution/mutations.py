import copy
from enum import Enum
from random import random

from modules.block import Block, get_new_random_input_block
from modules.cell import Cell
from modules.model import Model
from search_space import FIRST_INPUT, SECOND_INPUT


class MutationType(Enum):
    """
        the mutation type
    """
    CHANGE_BLOCK_INPUT = 0
    SWAP_OPERATION = 1


def _mutate_block_input(b: Block) -> None:
    """
    Mutate one of the inputs of the block
    :return: void (None)
    """
    inp = random.choice([FIRST_INPUT, SECOND_INPUT])

    if inp == FIRST_INPUT:
        b.first_input_block = get_new_random_input_block(b.block_number)
    else:
        b.second_input_block = get_new_random_input_block(b.block_number)


def _mutate_block_op(b: Block):
    """
    Mutate the block operation
    :return: the
    """
    selected_input = random.randint(FIRST_INPUT, SECOND_INPUT)

    if selected_input == FIRST_INPUT:
        b.first_input_op = Block.get_random_operation()
    else:
        b.second_input_op = Block.get_random_operation()


def mutate_block(b: Block) -> None:
    """
    Mutate this block by changing it's inputs or swapping it's operation
    :return: void (None)
    """
    mutation_type = random.choice(list(MutationType))

    if mutation_type == MutationType.CHANGE_BLOCK_INPUT:
        _mutate_block_input(b)
    else:
        _mutate_block_op(b)


def _get_random_block(cell: Cell) -> Block:
    """
    get a block randomly
    :return: the block that was selected
    """
    return random.choice(cell.blocks)


def mutate_cell(cell: Cell):
    """
    mutate the cell (in place)
    :return: the mutated cell
    """
    if len(cell.blocks) == 0:
        raise RuntimeError('Cannot mutate empty block list.')

    b = _get_random_block(cell)
    mutate_block(b)


def mutate_model(model: Model) -> Model:
    """
    Mutates this model by mutation of either one of the cells and returns a new model with the specific mutation
    :return: the modified model
    """
    new_model = Model()
    new_model.normal_cell = copy.deepcopy(model.normal_cell)
    new_model.reduction_cell = copy.deepcopy(model.reduction_cell)

    cell_to_mutate: Cell = random.choice([new_model.normal_cell, new_model.reduction_cell])
    mutate_cell(cell_to_mutate)

    return new_model
