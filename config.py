#!/usr/bin/env python
import logging

# 1. Backbone network parameters
IN_CHANNELS = 5
OUT_CHANNELS = 64
C = 20

# 2. EM parameters
K_MAX = 100           # for training
K_MAX_VAL = 300       # for validation
ITER_EM = 5

# 3. Scaling factors for generaing XYLab
COLOR_SCALE = 0.26
P_SCALE = 0.40

# 4. Training-related parameters
TRAIN_H = 201       # for training
TRAIN_W = 201       # for training
NUM_CLASSES = 50
BATCH_SIZE = 8
LR = 0.0001
ITER_MAX = 500000
ITER_TEST = 1000
ITER_SAVE = 10000

# 5. Hardware-related parameters
RAND_SEED = 2356
DEVICE = 0
DEVICES = list(range(0, 4))
NUM_WORKERS = 4

# 6. Dataset, log, model directories
ROOT = 'path/to/BSDS500'
LOG_DIR = 'path/to/logdir'
MODEL_DIR = 'path/to/models'

# 7. Logger
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
#logger.addHandler(ch)
