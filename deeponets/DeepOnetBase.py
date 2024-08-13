import os
import torch
import torch.nn as nn
import time
import pickle
import numpy as np
from scipy.interpolate import interp2d, griddata, interpn
from torch.utils.data import Dataset, DataLoader
from abc import  abstractmethod
from datasets.DONDataset import DONDataset
from config import params as args
import utils as doputils
from config import DEVICE


class DeepOnetBase(nn.Module):
    def __init__(self, nodes, permutation_indices, feature_norms, num_branch_inputs=2, num_basis_functions=240):
        super(DeepOnetBase, self).__init__()

        self.nodes                  = nodes
        self.permutation_indices    = permutation_indices
        self.trunk_inputs           = None

        self.generate_trunk_inputs()

        self.relu_activation = nn.ReLU()
        self.tanh_activation = nn.Tanh()

        self.num_basis_functions    = num_basis_functions   
        self.num_branch_inputs      = num_branch_inputs

        self.reshape_feature_tensor = None
        self.dataset_type           = DONDataset

        self.feature_norms = []
        for f_norms in feature_norms: 

            if type(f_norms) is list:
                local_norms = []
                for element in f_norms: 
                    local_norms.append(doputils.to_torch(element, args.dtype))

                self.feature_norms.append(local_norms)
            else:
                self.feature_norms.append(doputils.to_torch(f_norms, args.dtype))


    def _forward_branch(self, inputs):
        return self.branch_net(inputs)


    def init_params(self):
        self.apply(self._init_layers)    

    
    def _init_layers(self, m):        
        pass

    def generate_trunk_inputs(self, inputs=None):
        if(inputs is None):
            inputs = self.nodes

        self.trunk_inputs = torch.tensor(   inputs, 
                                            requires_grad=True, 
                                            dtype=args.dtype, device=DEVICE)

        self.trunk_inputs_don = self.trunk_inputs
        
        return self.trunk_inputs



    def forward_trunk(self, inputs=None):
        if inputs is None:
            trunk_out = self.trunk_net(self.trunk_inputs)
            if(args.BC_trunk):
                trunk_out = self.remove_bc(self.trunk_inputs, trunk_out)
        else:
            trunk_out = self.trunk_net(inputs)
            if(args.BC_trunk):
                trunk_out = self.remove_bc(inputs, trunk_out)

        return trunk_out
            

    @abstractmethod
    def remove_bc(self, trunk_inputs=None, trunk_outputs=None):
        raise NotImplementedError

    @abstractmethod
    def forward(self, branch_inputs, trunk_inputs=None):
        raise NotImplementedError


    @abstractmethod
    def infer(self, input_features_don, fe_problem_target):
        raise NotImplementedError



