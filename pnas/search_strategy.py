import os
import sys
import time
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchsummary import summary

import config
import search_space
from cifar10_loader import get_cifar10_sets
from config import device
from modules.cell import Cell
from modules.model import Model
from pnas import pnas_utilities
from pnas.pnas_utilities import _expand_cells, _get_normal_and_reduction_cells, PNASDataset, _prediction_accuracy, \
    _to_architecture_tensor
from pnas.surrogate_function import Surrogate

RATIO = 0.8
EPOCHS = 0
SURROGATE_BATCH_SIZE = 32
SURROGATE_EPOCHS = 250

CHILD_LR_MAX = 1e-5
CHILD_L2_REG = 3e-4

SURROGATE_LAYERS = 1
SURROGATE_HIDDEN_SIZE = 64
SURROGATE_DROPOUT = 0
SURROGATE_MLP_DROPOUT = 0.1
SURROGATE_MLP_LAYERS = 1
SURROGATE_MLP_HIDDEN_LAYERS = 2
SURROGATE_L2_REG = 0
SURROGATE_LR = 0.001
SURROGATE_GRAD_BOUND = 5.0

criterion = nn.CrossEntropyLoss()

train_loader, test_loader, classes = get_cifar10_sets()

OUTPUT_DIRECTORY = 'output'


def train_and_evaluate_network(net: Model) -> float:
    """

    :param net:
    :return:
    """
    net.setup_modules()
    net.to(device)

    summary(net, (3, 32, 32))

    train_network(net)
    return evaluate_architecture(net)


def train_network(net: nn.Module):
    """
    Train the network on cifar 10
    :param net: the network
    :return:
    """
    optimizer = optim.SGD(net.parameters(), lr=CHILD_LR_MAX, momentum=0.9, weight_decay=CHILD_L2_REG)
    start = time.perf_counter()

    print('Training architecture for {} epochs.'.format(EPOCHS))
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # inputs, labels = data[0], data[1]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 64 == 63:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    end = time.perf_counter()
    print('Finished Training. It took: {}'.format((end - start)))


def evaluate_architecture(net: Model) -> float:
    """
    evaluate the architecture performance
    :param net: the network
    :return: accuracy
    """
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total

    print("---")

    normal_cell_str = net.normal_cell.__str__()
    reduction_cell_str = net.reduction_cell.__str__()

    print('Normal: [{}], Reduction: [{}]'.format(normal_cell_str, reduction_cell_str))
    print('The network achieved the following accuracy: {}'.format(accuracy))

    with open('{}/architectures.csv'.format(OUTPUT_DIRECTORY), 'a+') as fd:
        fd.write('{} {},{}'.format(normal_cell_str, reduction_cell_str, accuracy))

    print("---")

    return accuracy


def _build_datasets(models: List[List[Model]], targets: List[List[float]]) -> (DataLoader, DataLoader):
    """
    build a training and a validation data set
    :param models:
    :param targets:
    :return:
    """
    assert len(models) == len(targets)

    model_accuracies = []

    for i in range(len(models)):
        model_accuracies.extend(
            zip([_to_architecture_tensor(m) for m in models[i]], targets[i]))

    np.random.shuffle(model_accuracies)

    split = int(len(model_accuracies) * RATIO)

    train_model_accuracies = model_accuracies[:split]
    train_model = [m_a[0] for m_a in train_model_accuracies]
    train_accuracies = [m_a[1] for m_a in train_model_accuracies]

    validation_model_accuracies = model_accuracies[split:]
    validation_model = [m_a[0] for m_a in validation_model_accuracies]
    validation_accuracies = [m_a[1] for m_a in validation_model_accuracies]

    train_dataset = PNASDataset(train_model, train_accuracies)
    validation_dataset = PNASDataset(validation_model, validation_accuracies, train=False)

    print("Surrogate train={}, valid={}".format(len(train_dataset.inputs), len(validation_dataset.inputs)))

    train_queue = torch.utils.data.DataLoader(
        train_dataset, batch_size=SURROGATE_BATCH_SIZE, shuffle=True, pin_memory=True)
    validation_queue = torch.utils.data.DataLoader(
        validation_dataset, batch_size=SURROGATE_BATCH_SIZE, shuffle=True, pin_memory=True)

    return train_queue, validation_queue


def _validate_surrogate_function(model: Surrogate, validate_queue: DataLoader) -> float:
    """
    validate the surrogate function on unknown cells
    :param model:
    :param validate_queue:
    :return: the accuracy
    """

    targets = []
    predictions = []

    with torch.no_grad():
        model.eval()
        for step, sample in enumerate(validate_queue):
            surrogate_input = sample['surrogate_input'].to(device)
            surrogate_target = sample['surrogate_target'].to(device)

            predict_value = model(surrogate_input)

            targets.append(surrogate_target)
            predictions.append(predict_value)

    return _prediction_accuracy(targets, predictions)


def _train_surrogate_function(model: Surrogate, models: List[List[Model]], accuracies: List[List[float]],
                              optimizer: torch.optim.Adam) -> None:
    """

    :param model:
    :return:
    """
    train_queue, validate_queue = _build_datasets(models, accuracies)
    loss = sys.maxsize

    for epoch in range(SURROGATE_EPOCHS):
        model.train()

        for step, sample in enumerate(train_queue):
            surrogate_input = sample['surrogate_input'].to(device)
            surrogate_target = sample['surrogate_target'].to(device)

            optimizer.zero_grad()
            predict_value = model(surrogate_input)

            loss = F.mse_loss(predict_value.squeeze(), surrogate_target.squeeze())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), SURROGATE_GRAD_BOUND)

            optimizer.step()

        print('Ep {} Surrogate loss: {}'.format(epoch, loss))

        if epoch % 100 == 0:
            acc = _validate_surrogate_function(model, validate_queue)
            print("Ep {} Validation accuracy: {}".format(epoch, acc))


def _surrogate_infer(surrogate: Surrogate, models: List[Model]) -> List[float]:
    """

    :param model:
    :param models:
    :return:
    """
    surrogate.eval()
    acc = []

    model_inputs = PNASDataset([_to_architecture_tensor(model) for model in models], train=False)
    infer_set = torch.utils.data.DataLoader(
        model_inputs, batch_size=SURROGATE_BATCH_SIZE, shuffle=False, pin_memory=True)

    for step, sample in enumerate(infer_set):
        surrogate_in = sample['surrogate_input'].to(device)
        acc = surrogate(surrogate_in)

        print(acc)
        print(acc.shape)

        acc = acc.view(-1)

        print(acc)
        print(acc.shape)

    return acc


def _cell_combinations_to_cnn(cell_combinations: List[Tuple[Cell, Cell]]) -> List[Model]:
    """
    Build a model from the cell combination and return it
    :param cell_combinations: the cell combinations
    :return: the models
    """
    models = []

    for normal_cell, reduction_cell in cell_combinations:
        m = Model()
        m.normal_cell = normal_cell
        m.reduction_cell = reduction_cell

        models.append(m)

    return models


def _normalize_accuracy(accuracies: List[float]) -> List[float]:
    """

    :param accuracies:
    :return:
    """
    min_acc = min(accuracies)
    max_acc = max(accuracies)
    return [(acc - min_acc) / (max_acc - min_acc) for acc in accuracies]


def _get_best_models(cells: List[Cell], models: List[Model], accuracies: List[float], beam_size: int) -> List[
    Tuple[Cell, Model, float]]:
    """

    :param models:
    :param accuracies:
    :param beam_size:
    :return:
    """
    assert len(cells) == len(models) == len(accuracies), 'len(cells)={}, len(models)={}, len(accuracies)={}'.format(
        len(cells), len(models), len(accuracies))
    models_with_acc = sorted(zip(cells, models, accuracies), key=lambda x: x[-1], reverse=True)

    return models_with_acc[:beam_size]


def progressive_neural_architecture_search(max_num_blocks: int, max_epochs: int, number_of_filters_in_first_layer: int,
                                           beam_size: int, normal_cells_per_stack: int, stack_count: int) -> List[
    Tuple[Cell, float]]:
    search_space.NUMBER_OF_BLOCKS_PER_CELL = 1
    pnas_utilities.MAX_NUMBER_OF_BLOCKS_PER_CELL = max_num_blocks
    search_space.STACK_COUNT = stack_count
    search_space.NUMBER_OF_NORMAL_CELLS_PER_STACK = normal_cells_per_stack
    search_space.NUMBER_OF_FILTERS = number_of_filters_in_first_layer

    global EPOCHS
    EPOCHS = max_epochs

    cells = [_expand_cells([])]
    cells[0] = cells[0][:4]
    normal_and_reduction_cell_combinations = _get_normal_and_reduction_cells(cells[0])

    models = []
    accuracies = []
    targets = []

    models.append(_cell_combinations_to_cnn(normal_and_reduction_cell_combinations))
    accuracies.append([train_and_evaluate_network(m) for m in models[0]])
    targets.append(_normalize_accuracy(accuracies[0]))

    surrogate = Surrogate(SURROGATE_HIDDEN_SIZE, SURROGATE_LAYERS, SURROGATE_DROPOUT, SURROGATE_MLP_DROPOUT,
                          SURROGATE_MLP_LAYERS,
                          SURROGATE_MLP_HIDDEN_LAYERS)
    surrogate.to(device)

    surrogate_optimizer = torch.optim.Adam(surrogate.parameters(), lr=SURROGATE_LR, weight_decay=SURROGATE_L2_REG)

    print("Training surrogate.")
    _train_surrogate_function(surrogate, models, targets, surrogate_optimizer)

    print("Increasing block_size now.")

    for block_size in range(1, max_num_blocks):
        search_space.NUMBER_OF_BLOCKS_PER_CELL = block_size

        print("Beginning block_size={}".format(block_size + 1))
        cells.append(_expand_cells(cells[block_size - 1]))

        print("After expanding, there are {} cells.".format(len(cells[block_size])))

        normal_and_reduction_cell_combinations = _get_normal_and_reduction_cells(cells[block_size])

        models.append(_cell_combinations_to_cnn(normal_and_reduction_cell_combinations))
        predictions = _surrogate_infer(surrogate, models[block_size])

        remaining_models = _get_best_models(cells[block_size], models[block_size], predictions, beam_size)
        print("Reduced to {} models according to their accuracy.".format(len(remaining_models)))

        print("Training these architectures in order to update the surrogate.")
        accuracies.append([train_and_evaluate_network(m) for m in models[block_size]])
        targets.append(_normalize_accuracy(accuracies[block_size]))

        print("Updating predictor")
        _train_surrogate_function(surrogate, models, targets, surrogate_optimizer)

        print("End of block_size={}".format(block_size + 1))

    return list(
        map(lambda cell, _, acc: (cell, acc), _get_best_models(cells[-1], models[-1], accuracies[-1], beam_size)))


def main():
    config.OUTPUT_DIRECTORY = 'output'

    if not os.path.exists(config.OUTPUT_DIRECTORY):
        os.makedirs(config.OUTPUT_DIRECTORY)

    print("Performing PNAS.")
    cell_accs = progressive_neural_architecture_search(5, 1, 32, search_space.NUMBER_OF_FILTERS,
                                                       search_space.NUMBER_OF_NORMAL_CELLS_PER_STACK,
                                                       search_space.STACK_COUNT)

    print("We've found the following architectures:")

    for c, acc in cell_accs:
        print("----")
        print("Normal/Reduction: []".format(c.__str__()))
        print("Top-1: {:5.f}".format(acc))
        print("----")


if __name__ == '__main__':
    main()
