import argparse
import os
from typing import List, Optional

from modules.block import Block
from modules.cell import Cell
from search_space import Operation, GRAPH_OUTPUT_DIR


def _get_operation(nao_operation_number: int) -> Operation:
    """
    convert nao's operation index into our representation
    :param nao_operation_number: nao's index
    :return:
    """
    if nao_operation_number == 0:
        return Operation.CONV_SEP_3x3
    elif nao_operation_number == 1:
        return Operation.CONV_SEP_5x5
    elif nao_operation_number == 2:
        return Operation.AVG_POOL_3x3
    elif nao_operation_number == 3:
        return Operation.MAX_POOL_3x3
    else:
        return Operation.IDENTITY


def _parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('architecture_file', help='The path to your architecture file (usually named arch_pool.final',
                        type=str)

    return vars(parser.parse_args())


def _read_lines(file_path: str) -> List[str]:
    """
    read a file into a list of strings
    """
    with open(file_path) as f:
        content = f.readlines()
    return [x.strip() for x in content]


def _recover_cell(cell_string: str) -> Optional[Cell]:
    """
    Recover the cell from it's string representation
    :param cell_string: the cell string
    :return: the cell
    """
    token = list(filter(lambda x: x != '', cell_string.split(' ')))
    assert len(token) % 4 == 0

    cell = Cell()

    for i in range(0, len(token), 4):
        input_1, op_1, input_2, op_2 = int(token[i]), int(token[i + 1]), int(token[i + 2]), int(token[i + 3])

        b = Block(int(0.25 * i + 2))

        assert input_1 < b.block_number
        assert input_2 < b.block_number

        b.first_input_block = input_1
        b.second_input_block = input_2

        b.first_input_op = _get_operation(op_1)
        b.second_input_op = _get_operation(op_2)

        cell.blocks.append(b)

    return cell


def _recover_cells(architecture_string: str) -> (Cell, Cell):
    """
    Recover the normal and reduction cell representation from their strings
    :param architecture_string: the architecture string
    :return: Tuple containing the cell representations (normal, reduction_cell) or None on parse error
    """
    n = len(architecture_string)
    normal_cell_string = architecture_string[:n // 2]
    reduction_cell_string = architecture_string[n // 2:]

    normal_cell = _recover_cell(normal_cell_string)
    reduction_cell = _recover_cell(reduction_cell_string)

    return normal_cell, reduction_cell


def main():
    args = _parse_args()

    architecture_file_path = args['architecture_file']

    if not os.path.exists(architecture_file_path) or not os.path.isfile(architecture_file_path):
        raise FileNotFoundError('File containing the architecture descriptions could not be found.')

    architectures = _read_lines(architecture_file_path)

    for i, arch in enumerate(architectures):
        normal_cell, reduction_cell = _recover_cells(arch)

        n_c = normal_cell.to_graph('Normal Cell {}'.format(i))
        r_c = reduction_cell.to_graph('Reduction Cell {}'.format(i))

        if not os.path.exists(GRAPH_OUTPUT_DIR):
            os.makedirs(GRAPH_OUTPUT_DIR)

        n_c.render('normal_cell_{}'.format(i), directory=GRAPH_OUTPUT_DIR)
        r_c.render('reduction_cell_{}'.format(i), directory=GRAPH_OUTPUT_DIR)

        print('Rendered graph for both cells of {}'.format(i))


if __name__ == '__main__':
    main()
