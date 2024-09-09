import torch
import numpy as np
from datetime import datetime


# region Data creation

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


NUM_OF_ELEMENTS = 30
NUM_OF_ELEMENTS_DON = 30
NUM_OF_CASES = 5000
PROBLEM = 'poisson'  # Option: 'poisson', 'helmholtz'
SHOW_DATA_CREATION_IMAGES = True
DIMENSIONS = 1

K_SIGMA = 0.3
K_L0 = 0.1 
K_0 = 1.0
K_MIN = 0.3
F_SIGMA = 1.0
F_L0 = 0.1

# endregion

# region Jacobian solver

SOLVER_METHOD = 'Jacobi' # Option: 'Jacobi', 'GS', 'CG', 'MG-J', 'MG-GS'
ITERATION_METHOD = 'Numerical_DeepONet_Hybrid'
# Options: 'Numerical', 'DeepONet', 'Numerical_DeepONet_Hybrid', 'Numerical_DeepONet_Single'
NUMERICAL_TO_DON_RATIO = 25
MODES_OF_INTEREST = [1, 5, 10]
NUM_OF_ITERATIONS = 400
NUM_OF_MG_V_CYCLE = 10
NUM_OF_LEVELS = 6
OMEGA = 2 / 3

# endregion

# region Training params
FORCE_RETRAIN = False
PERCENTAGE_TEST = 0.15
EPOCHS = 10000
BATCH_SIZE = 500
REDUNDANCY = 1
LR = 0.001

NORM_THRESHOLD = float('inf')

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

PLOT_INTERVAL = 500
PRINT_INTERVAL = 100
LOSS_FUNC = 'L2_abs'

MODEL_NAME = 'training_2022-05-30_1P'
DATASET_NAME = 'processed_data_1P.pkl'

