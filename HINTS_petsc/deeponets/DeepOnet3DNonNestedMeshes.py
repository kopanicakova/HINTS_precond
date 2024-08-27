import os
import torch
import torch.nn as nn
import time
import utils as doputils
import pickle
import numpy as np
import copy
from scipy.interpolate import interp2d, griddata, interpn
from torch.utils.data import Dataset, DataLoader
from datasets.DONDataset import DONDataset
from deeponets.DeepOnetBase import DeepOnetBase
from config import params as args
from config import DEVICE
from utils import logger 


try:
  from firedrake import Function, restrict, prolong, inject
except ImportError:
  print("Firedrake not found")



class DeepOnet3DNonNestedMeshes(DeepOnetBase):
    def __init__(self, nodes_don, nodes_fem, feature_norms, permutation_indices, num_branch_inputs=1, num_basis_functions=240, autoencoder=None):
        
        self.nodes_fem        = nodes_fem
        
        super(DeepOnet3DNonNestedMeshes, self).__init__(   nodes=nodes_don, permutation_indices=permutation_indices, feature_norms=feature_norms,\
                                                num_branch_inputs=num_branch_inputs, num_basis_functions=num_basis_functions)

        self.num_x_points = len(np.unique(self.nodes[:,0]))
        self.num_y_points = len(np.unique(self.nodes[:,1]))
        self.num_z_points = len(np.unique(self.nodes[:,2]))

        self.reshape_feature_tensor = [-1, self.num_x_points, self.num_y_points, self.num_z_points]

        if(args.dofs_don==32):
            num_of_conv_layers = 4
        elif(args.dofs_don==16):
            num_of_conv_layers = 3
        elif(args.dofs_don==8):
            num_of_conv_layers = 2
        else:
            logger.info("TODO::DeepOnet3DNonNestedMeshes:: Set num of conv. layers")


        layers_list = [nn.Conv3d(in_channels=self.num_branch_inputs,
                                 out_channels=20 + 20,
                                 kernel_size=(3, 3, 3),
                                 stride=2,
                                 dtype = args.dtype),
                        nn.ReLU()]


        for conv_layer_index in range(2, num_of_conv_layers + 1):
            layers_list.append(nn.Conv3d(in_channels=20 * (2 ** (conv_layer_index - 2)) + 20,
                                         out_channels=20 * (2 ** (conv_layer_index - 1)) + 20,
                                         kernel_size=(3, 3, 3),
                                         stride=2, 
                                         dtype=args.dtype)) # add max pool layer 

            layers_list.append(nn.ReLU())


        layers_list.append(nn.Flatten())
        layers_list.append(nn.Linear(20 * (2 ** (num_of_conv_layers - 1)) + 20, 80, dtype=args.dtype))
        layers_list.append(nn.ReLU()) 
        layers_list.append(nn.Linear(80, 80, dtype=args.dtype))
        layers_list.append(nn.ReLU()) 
        layers_list.append(nn.Linear(80, self.num_basis_functions, dtype=args.dtype))


        self.branch_net = nn.Sequential(*layers_list).to(DEVICE)       


        # Trunk Net
        self.trunk_net = nn.Sequential(nn.Linear(3, 80, dtype=args.dtype),
                                       nn.Tanh(), # swish, leaky relu
                                       nn.Linear(80, 80, dtype=args.dtype),
                                       nn.Tanh(), # swish, leaky relu
                                       nn.Linear(80, self.num_basis_functions, dtype=args.dtype),
                                       nn.Tanh()).to(DEVICE)


        self.dataset_type       = DONDataset



    def generate_trunk_inputs(self, inputs=None):
        if(inputs is None):
            inputs = self.nodes_fem

        self.trunk_inputs = torch.tensor(   inputs, 
                                            requires_grad=True, 
                                            dtype=args.dtype, device=DEVICE)


        self.trunk_inputs_don = torch.tensor(   self.nodes, 
                                                requires_grad=True, 
                                                dtype=args.dtype, device=DEVICE)

        return self.trunk_inputs


    # TODO:: modify based on example 
    def remove_bc(self, trunk_inputs=None, trunk_outputs=None):

        trunk_outputs   = trunk_outputs.T * (trunk_inputs[:,2] - 1.0)
        trunk_outputs   = trunk_outputs.T

        return trunk_outputs


    def eval_branch(self, branch_inputs):
        for idx in range(0, len(branch_inputs)): 
            if(len(self.feature_norms)>0):
                f_norm = self.feature_norms[idx].reshape(self.reshape_feature_tensor)
                branch_inputs[idx] = branch_inputs[idx]/f_norm

        if(len(branch_inputs)==1):
            branch_outputs_net = self._forward_branch(branch_inputs[0][0][:, None])  

        elif(len(branch_inputs)==2):

            if(len(branch_inputs[0].shape)==5):
                branch_input_net = torch.cat((branch_inputs[0][0][:, None], branch_inputs[1][0][:, None]), dim=1)
            elif(len(branch_inputs[0].shape)==3):
                branch_input_net = torch.cat((branch_inputs[0][0][None, None, :], branch_inputs[1][0][None, None, :]), dim=1)
            else:
                print("Unknown size of branch .... ")
                exit(0)

            # print("branch_inputs ", branch_input_net.shape)
            branch_outputs_net = self._forward_branch(branch_input_net)  

        else:
            logger.info("DeepOnet2D:: Add more options and scalings for branch net")
            exit(0)

        return  branch_outputs_net 



    def forward(self, branch_inputs, trunk_inputs):

        trunk_outputs       = self.forward_trunk(trunk_inputs)   
        branch_outputs_net  = self.eval_branch(branch_inputs)

        u_pred  = branch_outputs_net @ trunk_outputs.T     # (num_cases, num_points)

        return u_pred


    # TODO:: generalize
    def infer(self, input_features_don, fe_problem_target, precomp_trunk=None):

        torch_features_don = []
        for s in range(len(input_features_don)):

            reshaped_features = input_features_don[s][self.permutation_indices]
            reshaped_features = reshaped_features.reshape(self.reshape_feature_tensor)
            reshaped_features = reshaped_features[None, :, :, :]
            
            torch_features_don.append(doputils.to_torch(reshaped_features, args.dtype))

        if(precomp_trunk is None):
            trunk_inputs    = torch.tensor(fe_problem_target.nodes, requires_grad=True,  dtype=args.dtype, device=DEVICE)
            u_pred          = self.forward(torch_features_don, trunk_inputs)
        else:
            branch_outputs_net  = self.eval_branch(torch_features_don)
            u_pred  = branch_outputs_net @ precomp_trunk.T  

        return u_pred.detach().numpy()
