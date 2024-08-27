import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from config import params as args
import utils
import copy
from utils import logger 

try:
  from petsc_solvers.PCHelpers import * 
except ImportError:
  print("Firedrake not found")



class DONDataset(Dataset):
  def __init__(self, data_u, data_features, permutation_indices, reshape_feature_tensor=None, transform=None):
    self.initialize(data_u, data_features, permutation_indices, reshape_feature_tensor, transform)


  def initialize(self, data_u, data_features, permutation_indices, reshape_feature_tensor, transform):
    if(reshape_feature_tensor is not None):
        self.u = data_u
    else:
        self.u = data_u

    self.features = data_features

    for fs in range(len(self.features)):
        self.features[fs] = self.features[fs][:,permutation_indices]

        if(reshape_feature_tensor is not None):
            self.features[fs] = self.features[fs].reshape(reshape_feature_tensor)


  def __len__(self):
    return self.u.shape[0]


  def __getitem__(self, idx):
    f_batch=[]
    for fs in range(len(self.features)):
        f_batch.append(utils.to_torch(self.features[fs][idx], dtype=args.dtype))    
        
    u_batch = utils.to_torch(self.u[idx], dtype=args.dtype)

    return f_batch, u_batch



