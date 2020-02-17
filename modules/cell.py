import random

import torch
from graphviz import Digraph
from torch import nn

from modules.block import Block
from search_space import NUMBER_OF_BLOCKS_PER_CELL
from utilities import _pad_tensor


class Cell(nn.Module):
    """
        Either a reduction or a normal cell. The cell is a building block of our neural network architecture.
        A cell is a directed acyclic graph, where each node applies an operation to an input or a previous node.

        We are using
    """

    def __init__(self):
        """
            Initializes a new instance of this cell
        """
        super(Cell, self).__init__()
        self.blocks: nn.ModuleList = nn.ModuleList([])

    def _get_random_block(self) -> Block:
        """
        get a block randomly
        :return: the block that was selected
        """
        return random.choice(self.blocks)

    def build_ops(self, stack_num: int, stack_pos: int, is_normal_cell: bool) -> nn.Module:
        """
        build all block ops
        :param is_normal_cell: whether the current cell is a normal cell
        :param stack_pos: the position of this block in the stack of cells
        :param stack_num: the stack number to which this block belongs, determines the channels of Conv2d
        :return: void
        """
        for i, block in enumerate(self.blocks):
            b: Block = block
            b.build_ops(stack_num, stack_pos, is_normal_cell, self.blocks[:i])

        return self

    def mutate(self):
        """
        mutate the cell
        :return:
        """
        if len(self.blocks) == 0:
            raise RuntimeError('Cannot mutate empty block list.')

        b = self._get_random_block()
        b.mutate()

    def forward(self, input_a: torch.tensor, input_b: torch.tensor):
        """
        define a forward pass
        :param input_a: the first input vector/tensor
        :param input_b: the second input vector/tensor
        :return: a tensor of all concatenated
        """

        tensors = [input_a, input_b]
        block_used_as_input = [0, 1]

        print('CELL. IN1-DIM: {}. IN2-DIM: {}'.format(input_a.shape, input_b.shape))

        for block in self.blocks:
            block_output = block(tensors[block.first_input_block], tensors[block.second_input_block])
            tensors.append(block_output)

            block_used_as_input += [block.first_input_block, block.second_input_block]

        output_blocks = set(range(NUMBER_OF_BLOCKS_PER_CELL + 2)) - set(block_used_as_input)

        if len(output_blocks) > 1:
            output_blocks_sorted = sorted(output_blocks, key=lambda x: tensors[x].shape, reverse=True)
            reference_tensor = tensors[output_blocks_sorted[0]]

            for t in output_blocks_sorted[1:]:
                assert reference_tensor.shape >= tensors[t].shape

                if reference_tensor.shape > tensors[t].shape:
                    tensors[t] = _pad_tensor(tensors[t], reference_tensor)

        out = tensors[list(output_blocks)[0]]

        for o_b in list(output_blocks)[1:]:
            out = out.add(tensors[o_b])

        print('CELL-OUT. OUT: {}'.format(out.shape))

        return out

    def to_graph(self, title: str) -> Digraph:
        """
        convert this block into a digraph
        :return: the digraph
        """
        graph = Digraph()
        graph.graph_attr['label'] = title

        block_color = 'chartreuse'
        op_color = 'cadetblue1'

        graph.node('block-0', 'Input 0', color=block_color, style='filled')
        graph.node('block-1', 'Input 1', color=block_color, style='filled')

        blocks_used_as_input = []

        for b in self.blocks:
            block: Block = b

            first_input_op_node = 'block-{}-op-{}'.format(block.block_number, 1)
            second_input_op_node = 'block-{}-op-{}'.format(block.block_number, 2)

            graph.node(first_input_op_node, block.first_input_op.__str__(), color=op_color, style='filled')
            graph.node(second_input_op_node, block.second_input_op.__str__(), color=op_color, style='filled')

            graph.edge('block-{}'.format(block.first_input_block), first_input_op_node)
            graph.edge('block-{}'.format(block.second_input_block), second_input_op_node)

            graph.node('block-{}'.format(block.block_number), 'Block-{} [cat]'.format(block.block_number),
                       color=block_color, style='filled')

            graph.edge(first_input_op_node, 'block-{}'.format(block.block_number))
            graph.edge(second_input_op_node, 'block-{}'.format(block.block_number))

            blocks_used_as_input += [block.first_input_block, block.second_input_block]

        outputs = list(set(range(len(self.blocks) + 2)) - set(blocks_used_as_input))

        graph.node('output', 'Output', color=block_color, style='filled')

        for output_block in outputs:
            graph.edge('block-{}'.format(output_block), 'output')

        return graph
