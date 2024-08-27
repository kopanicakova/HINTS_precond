from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import os
import torch
import subprocess
import argparse


# @staticmethod
def setup_distr_env():
    os.environ['MASTER_PORT'] = '29501'
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NNODES']
    os.environ['LOCAL_RANK'] = '0'
    os.environ['RANK'] = os.environ['SLURM_NODEID']
    node_list = os.environ['SLURM_NODELIST']
    master_node = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
    os.environ['MASTER_ADDR'] = master_node


# @staticmethod
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# @staticmethod
def get_params():
    parser = argparse.ArgumentParser(description='Params')

    parser.add_argument('--seed', default=1, type=int, help='seed')
    parser.add_argument('--dtype', default=torch.double, type=torch.dtype, help='dtype')

    parser.add_argument('--dofs_fem', default=32, type=int, help='number of dofs')
    parser.add_argument('--dofs_don', default=32, type=int, help='number of dofs')


    parser.add_argument('--mesh', default="", type=str, help='mesh')
    parser.add_argument('--mesh_ref_don', default=1, type=int, help='Refinement level DON')
    parser.add_argument('--mesh_ref_fem', default=1, type=int, help='Refinement level FEM')
    

    parser.add_argument('--num_samples_total', default=1000, type=int, help='num of samples ')
    parser.add_argument('--num_samples', default=1000, type=int, help='num of samples ')
    parser.add_argument('--redudancy', default=1, type=int, help='redudancy')
    parser.add_argument('--num_basis_functions', default=128, type=int, help='num of samples ')
    parser.add_argument('--BC_trunk', default=False, type=str2bool, nargs='?', help='apply BC to trunk?')


    parser.add_argument('--f_sigma', default=1.0, type=float, help='f-sigma')
    parser.add_argument('--f_lo', default=0.1, type=float, help='f-lo')

    
    parser.add_argument('--k_sigma', default=1.0, type=float, help='k-sigma')
    parser.add_argument('--k_lo', default=0.3, type=float, help='k-lo')    
    # parser.add_argument('--wave_number', default=10, type=int, help='wave_number')    
    # parser.add_argument('--k_o', default=1.0, type=float, help='k-o')    
    # parser.add_argument('--k_min', default=0.3, type=float, help='k-min')    

    
    # TODO:: add whole array 
    parser.add_argument('--res_sigma', default=1.0, type=float, help='k-sigma')
    parser.add_argument('--res_lo', default=1.0, type=float, help='k-lo')    
    parser.add_argument('--res_per_sample', default=1, type=int, help='res_per_sample')


    parser.add_argument('--percentage_test', default=0.15, type=float, help='percentage test')
    parser.add_argument('--epochs', default=100000, type=int, help='num epochs')
    parser.add_argument('--batch_size', default=5000, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--optimizer_type', default="adam", type=str, help='optimizer type')
    parser.add_argument('--max_stall', default=10000, type=int, help='terminate learning tol')
    parser.add_argument('--sampling_method', default=1, type=int, help='sampling method')


    parser.add_argument('--plot_interval', default=1000, type=int, help='plot interval')
    parser.add_argument('--print_interval', default=1000, type=int, help='print interval')


    parser.add_argument('--loss_function', default="L2_relative", type=str, help='type of loss function')

    
    parser.add_argument('--force_retrain', default=True, type=str2bool, nargs='?', help='force retrain')
    parser.add_argument('--recreate_data', default=True, type=str2bool, nargs='?', help='force retrain')

    parser.add_argument('--only_train', default=False, type=str2bool, nargs='?', help='force retrain')
    parser.add_argument('--only_generate_data', default=False, type=str2bool, nargs='?', help='force retrain')


    parser.add_argument('--model_name', default="poisson1D", type=str, help='model name')
    parser.add_argument('--model_name_load', default="none", type=str, help='model name')
    parser.add_argument('--dataset_name', default="poisson1D.pkl", type=str, help='dataset name')


    parser.add_argument('--csv_stat_name', default="csv_stats.csv", type=str, help='model name')


    args, unknown = parser.parse_known_args()


    return args    



# holds all parameters used for simulation
params = get_params()


res_norm0   = 1e9
time0       = 0.0


# GPU/CPU code? 
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")


