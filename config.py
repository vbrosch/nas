import torch

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
INPUT_DIM = (3, 28, 28)
NUM_EPOCHS = 10
VERBOSE = True
