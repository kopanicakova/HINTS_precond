from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import os
import torch
import subprocess
import argparse
from config import DEVICE
import logging
import numpy as np
from config import params as args
import pickle



def POD(y, num_modes):
    n = len(y)
    y_mean = np.mean(y, axis=0)
    y = y - y_mean
    C = 1 / (n - 1) * y.T @ y
    w, v = np.linalg.eigh(C)
    # w = np.flip(w)
    v = np.fliplr(v)
    v *= len(y_mean) ** 0.5

    

    # w_cumsum = np.cumsum(w)
    # print(w_cumsum[:16] / w_cumsum[-1])
    # plt.figure()
    # plt.plot(y_mean)
    # plt.figure()
    # for i in range(8):
    #     plt.subplot(2, 4, i + 1)
    #     plt.plot(v[:, i])
    # plt.show()
    return y_mean, v[:, :num_modes]




# @staticmethod
def create_folder(folder_path):
    if not os.path.exists(folder_path):
        try:
            os.mkdir(folder_path)
        except OSError as error:
            pass


# @staticmethod
def to_numpy(x):
    if isinstance(x, list):
        return [to_numpy(i) for i in x]
    else:
        return x.cpu().detach().numpy().astype('float64')



# @staticmethod
def to_torch(x, dtype=torch.float, shape=None):
    if(shape==None):
        return torch.tensor(x, dtype=dtype, device=DEVICE)
    else:
        t_tensor =  torch.tensor(x, dtype=dtype, device=DEVICE)
        return t_tensor.reshape(shape)



# @staticmethod
def cleanup_folder(folder): 
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))                        



def get_data(percentage_test, data_folder='data', force_create=False):
    data_path = os.path.join(data_folder, args.dataset_name)
    if os.path.exists(data_path) and not force_create:
        logger.info('Loading data from file')
        return pickle.load(open(data_path, 'rb'))
    else:
        logger.info('Please, create data in FE-problem sampler!')
        exit(0)     

    return data



logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s, %(levelname)s:     %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S')


logger = logging.getLogger()
logger.setLevel(15)

