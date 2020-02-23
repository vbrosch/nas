import copy
import itertools
from typing import List, Dict, Tuple

import torch

from modules.block import Block
from modules.cell import Cell
from modules.model import Model
from search_space import Operation

MAX_NUMBER_OF_BLOCKS_PER_CELL = 0


def _prediction_accuracy(inputs, targets) -> float:
    n = len(inputs)
    assert n == len(targets)

    mse = torch.nn.MSELoss()

    la_tensor = torch.FloatTensor(inputs)
    lb_tensor = torch.FloatTensor(targets)

    return mse(la_tensor, lb_tensor).mean().item()


def _get_input_word(input_num: int) -> str:
    """
    convert this input number into the input word in the dictionary
    :param input_num: the input num
    :return: the input word
    """
    return 'input-'.format(input_num)


def _get_op_word(op: Operation) -> str:
    """
    get the operation identifier in the dictionary
    :param op: the operation
    :return: the operation identifier
    """
    return 'op-{}'.format(op.__str__())


def _get_vocabulary_size() -> int:
    """
    get the vocabulary size
    :return: the size of the vocabulary for the embeddings
    """
    return len(_get_vocabulary())


def _get_vocabulary() -> Dict[str, int]:
    """
    get the full vocabulary
    :return: the vocabulary
    """
    vocabulary = [_get_input_word(i) for i in range(MAX_NUMBER_OF_BLOCKS_PER_CELL + 2)]
    vocabulary += [_get_op_word(op) for op in list(Operation)]

    return {w: i for i, w in enumerate(vocabulary)}


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
        expanded_cells.append(tmp_cell)

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


def _get_normal_and_reduction_cells(cells: List[Cell]) -> List[Tuple[Cell, Cell]]:
    """
    get a pairwise combination of normal and reduction cells
    :param cells: the cells
    :return: normal and reduction cell combination
    """
    return [(cell, cell) for cell in cells]


def _to_architecture_tensor(model: Model) -> List[int]:
    """

    :return:
    """
    normal_cell_tokens = model.normal_cell.__str__().split(' ')
    reduction_cell_tokens = model.normal_cell.__str__().split(' ')

    architecture_tokens = normal_cell_tokens + reduction_cell_tokens

    tokens = [int(t) for t in architecture_tokens]

    return tokens


class PNASDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=None, train=True):
        super(PNASDataset, self).__init__()
        if targets is not None:
            assert len(inputs) == len(targets)
        self.inputs = copy.deepcopy(inputs)
        self.targets = copy.deepcopy(targets)
        self.train = train

    def __getitem__(self, index):
        surrogate_input = self.inputs[index]
        surrogate_target = None
        if self.targets is not None:
            surrogate_target = [self.targets[index]]
        if self.train:
            sample = {
                'surrogate_input': torch.LongTensor(surrogate_input),
                'surrogate_target': torch.FloatTensor(surrogate_target)
            }
        else:
            sample = {
                'surrogate_input': torch.LongTensor(surrogate_input)
            }
            if surrogate_target is not None:
                sample['surrogate_target'] = torch.FloatTensor(surrogate_target)
        return sample

    def __len__(self):
        return len(self.inputs)
