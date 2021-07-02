import torch
from math import log2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LRN_RATE = 1E-3
BATCH_SIZES = [16, 16, 16, 16, 16, 16, 16, 8, 4]
PIN_MEMORY = True

START_TRAINING_AT_IMG_SIZE = 4
IMG_SIZE = 512
IMG_CHANNELS = 3

Z_DIM = 256
IN_CHANNELS = 256

LAMBDA_GP = 10
NUM_STEPS = int(log2(IMG_SIZE / 4)) + 1

PROGRESSIVE_EPOCHS = [10] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKER = 8

SAVE_MODEL = True
LOAD_MODEL = True

ADAM_BETAS = (0.0, 0.99)
