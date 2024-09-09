import torch
import numpy as np
from datetime import datetime


# region Data creation

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


NUM_OF_ELEMENTS = 30  # if not equal to NUM_OF_ELEMENTS, needs interpolation for DeepONet branch input
NUM_OF_ELEMENTS_DON = 30
NUM_OF_CASES = 5000
PROBLEM = 'helmholtz'  # Option: 'poisson', 'helmholtz'
SHOW_DATA_CREATION_IMAGES = True
DIMENSIONS = 1

K_SIGMA = 2.0
K_L0 = 0.2
K0 = 8.0
K_MIN = 3.0
F_SIGMA = 1.0
F_L0 = 0.1

if PROBLEM == 'helmholtz':
    K_L0 = 0.3     # first:0.2
    F_SIGMA = 1.0
    F_L0 = 0.1
    if DIMENSIONS == 1:
        K_MIN = 3.0
        K_SIGMA = 2.0
        K_0 = 1.0 * 8.0
    if DIMENSIONS == 2:
        K_MIN = 3  # 6.0
        K_SIGMA = 0.5  # 0.3
        K_0 = 6  # 1.0 * 12.0
    if DIMENSIONS == 3:
        K_MIN = 3.0
        K_SIGMA = 0.2  # 0.5
        K_0 = 1.0 * 6.0

# endregion

# region Jacobian solver

SOLVER_METHOD = 'Jacobi' # Option: 'Jacobi', 'GS', 'CG', 'MG-J', 'MG-GS'
ITERATION_METHOD = 'Numerical_DeepONet_Hybrid'
# Options: 'Numerical', 'DeepONet', 'Numerical_DeepONet_Hybrid', 'Numerical_DeepONet_Single'
NUMERICAL_TO_DON_RATIO = 15
MODES_OF_INTEREST = [1, 5, 10] # list(range(1, 102, 10))  # [1, 31, 61]
NUM_OF_ITERATIONS = 600
NUM_OF_MG_V_CYCLE = 10
NUM_OF_LEVELS = 3
L_SHAPED = False
if L_SHAPED:
    LIFTING = False
    L_SHAPED_N_X = 21
    L_SHAPED_N_Y = 21
    L_SHAPED_LEVELS = 8
    NUM_OF_ELEMENTS = NUM_OF_ELEMENTS_DON  # TODO: put None and fix all places
if SOLVER_METHOD in ['Jacobi', 'MG-J']:
    if DIMENSIONS == 1:
        OMEGA = 2 / 3
    elif DIMENSIONS == 2:
        OMEGA = 4 / 5
    elif DIMENSIONS == 3:
        OMEGA = 6 / 7
else:
    OMEGA = 1

if DIMENSIONS == 2:
    NUM_OF_ELEMENTS_X = NUM_OF_ELEMENTS
    NUM_OF_ELEMENTS_Y = NUM_OF_ELEMENTS
    NUM_OF_ELEMENTS_DON_X = NUM_OF_ELEMENTS_DON
    NUM_OF_ELEMENTS_DON_Y = NUM_OF_ELEMENTS_DON
elif DIMENSIONS == 3:
    NUM_OF_ELEMENTS_X = NUM_OF_ELEMENTS
    NUM_OF_ELEMENTS_Y = NUM_OF_ELEMENTS
    NUM_OF_ELEMENTS_Z = NUM_OF_ELEMENTS
    NUM_OF_ELEMENTS_DON_X = NUM_OF_ELEMENTS_DON
    NUM_OF_ELEMENTS_DON_Y = NUM_OF_ELEMENTS_DON
    NUM_OF_ELEMENTS_DON_Z = NUM_OF_ELEMENTS_DON

# endregion

# region Training params
FORCE_RETRAIN = False
PERCENTAGE_TEST = 0.15
EPOCHS = 10000
BATCH_SIZE = 500
REDUNDANCY = 2
LR = 0.001


if DIMENSIONS == 2:
    LR = 0.0005
elif DIMENSIONS == 3:
    LR == 0.0001

NORM_THRESHOLD = 1
if L_SHAPED:
    NORM_THRESHOLD = 10

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")
PLOT_INTERVAL = 500
PRINT_INTERVAL = 100

LOSS_FUNC = 'L2_btw'
if PROBLEM == 'helmholtz' and DIMENSIONS == 1:
    LOSS_FUNC = 'L2_between'

# MODEL_NAME = 'training_' + str(datetime.now())[:10] + '_l_shaped'
if DIMENSIONS == 1:
	MODEL_NAME = 'training_2022-06-29_1H'
	DATASET_NAME = 'processed_data_1H.pkl'
if DIMENSIONS == 2:
    if L_SHAPED:
        MODEL_NAME = 'training_2022-07-27_l_shaped'
        DATASET_NAME = 'processed_data_l.pkl'
    else:
        MODEL_NAME = 'training_2022-08-03_2d_new'
        # MODEL_NAME = 'training_' + str(datetime.now())[:10] + '_2d'
        DATASET_NAME = 'processed_data_2d_new.pkl'
elif DIMENSIONS == 3:
    MODEL_NAME = 'training_2022-07-08_3d'
    DATASET_NAME = 'processed_data_3d_new.pkl'
# endregion


