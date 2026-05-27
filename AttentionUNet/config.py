import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ISIC2018", "data")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

INPUT_SIZE = 256
BATCH_SIZE = 8
NUM_EPOCHS = 60
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
USE_AMP = True
NUM_WORKERS = 4
SEED = 42

# U-Net base channels
BASE_CH = 64
