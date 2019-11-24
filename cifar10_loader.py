########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.
import torch
import torchvision
from torchvision import transforms


def get_cifar10_sets() -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader, tuple):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    training_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
    training_set_loader = torch.utils.data.DataLoader(training_set, batch_size=4,
                                                      shuffle=True, num_workers=2)

    testing_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
    testing_set_loader = torch.utils.data.DataLoader(testing_set, batch_size=4,
                                                     shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return training_set_loader, testing_set_loader, classes
