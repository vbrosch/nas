import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = (3, 28, 28)
NUM_EPOCHS = 25
