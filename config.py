import torch


MAX_LEN = 128
EMBED_DIM = 128
NHEAD = 2
FFN_HID_DIM = 128
NUM_LAYERS = 4
BATCH_SIZE = 16
EPOCHS = 15
LR = 3e-4
num_classes = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')