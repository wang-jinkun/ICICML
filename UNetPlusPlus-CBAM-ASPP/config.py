import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ISIC2018", "data")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

INPUT_SIZE = 256
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
USE_AMP = True
NUM_WORKERS = 4
SEED = 42

# U-Net++ deep supervision weights for L1, L2, L3, L4 outputs
DS_WEIGHTS = [0.125, 0.25, 0.25, 0.375]
WARMUP_EPOCHS = 5  # linear warmup from 0 to LEARNING_RATE
