try:
  import sys, petsc4py
  petsc4py.init(sys.argv)
  from petsc4py import PETSc
except ImportError:
  print("Petsc4py not found")

import argparse
import os
import numpy as np
import time
import random
import torch
import utils as doputils
import matplotlib 

from deeponets.DeepOnet3DNonNestedMeshes    import DeepOnet3DNonNestedMeshes
from datasets.Helmholtz3D_cylinder          import Helmholtz3D_cylinder
from trainers.Trainer                       import Trainer
from datasets.LinearProblemFiredrakeSampler import FEMtoDONTranfer

import matplotlib.pyplot as plt
from utils import logger
import pandas as pd
from config import params as args

try:
  from firedrake import UnitIntervalMesh, Function, MeshHierarchy, COMM_WORLD, FunctionSpace, Interpolator
  from firedrake import prolong, NonlinearVariationalSolver, File, NonlinearVariationalProblem, assemble
  from petsc_solvers.PCHelpers import * 
except ImportError:
  print("Firedrake not found")


import warnings
warnings.filterwarnings("ignore")  


def preprocess_data_to_fit_DON_size(): 

    data_generator  = problem_type(mesh_path =" mesh/cylinder_coarse.e", sample_k=False, k_value=6, ncells=args.dofs_don)
    data            = data_generator.get_data(  percentage_test=args.percentage_test,
                                                force_create=args.recreate_data)

    don_to_fem_transfer = FEMtoDONTranfer(data_generator.mesh, data_generator.don_mesh)

    ########################################################################################################
    features_don_train_new = []
    for f_id  in range(0, len(data['features_fem_train'])):
      zeros_new  = np.zeros((data['features_fem_train'][f_id].shape[0], int((args.dofs_don+1)**3)))
      features_don_train_new.append(zeros_new)


    features_don_test_new = []
    for f_id  in range(0, len(data['features_fem_test'])):
      zeros_new  = np.zeros((data['features_fem_test'][f_id].shape[0], int((args.dofs_don+1)**3)))
      features_don_test_new.append(zeros_new)      


    ########################################################################################################
    for f_id  in range(0, len(data['features_fem_train'])):
        for i in range(data['features_fem_train'][f_id].shape[0]):
          don_to_fem_transfer.k_func.dat.data[:]  = data['features_fem_train'][f_id][i, :]
          don_to_fem_transfer.k_func_target       = assemble(don_to_fem_transfer.interpolate())
          features_don_train_new[f_id][i, :]      = don_to_fem_transfer.k_func_target.dat.data[:]


    for f_id  in range(0, len(data['features_fem_test'])):
            for i in range(data['features_fem_test'][f_id].shape[0]):
              don_to_fem_transfer.k_func.dat.data[:]  = data['features_fem_test'][f_id][i, :]
              don_to_fem_transfer.k_func_target       = assemble(don_to_fem_transfer.interpolate())
              features_don_test_new[f_id][i, :]      = don_to_fem_transfer.k_func_target.dat.data[:]



    ########################################################################################################
    data_new = {  'nodes_fem': data.get('nodes_fem'),
                  'nodes_don': data_generator.nodes_don,
                  'feature_norms': data["feature_norms"],
                  'permutation_indices_don': data_generator.permutation_indices_don,
                  'u_train': data["u_train"],
                  'u_test': data["u_test"],
                  'features_fem_train': data["features_fem_train"],
                  'features_fem_test': data["features_fem_test"],
                  'features_don_train': features_don_train_new,
                  'features_don_test': features_don_test_new,}



    args.dataset_name = args.dataset_name + "_DON_size_"+str(args.dofs_don)
    data_generator.save(data_new)



def train_net(problem_type, num_basis_functions):

    # data_generator  = problem_type(mesh_path =" mesh/cylinder_coarse.e", sample_k=True)
    data_generator  = problem_type(mesh_path =" mesh/cylinder_coarse.e", sample_k=False, k_value=6)
    data            = data_generator.get_data(  percentage_test=args.percentage_test,
                                                force_create=args.recreate_data)

    path = 'outputs'
    os.makedirs(path, exist_ok=True)
    model_path = os.path.join('models', args.model_name)
    

    if(args.only_generate_data == False):    
      deeponet = DeepOnet3DNonNestedMeshes(   nodes_don = data['nodes_don'],
                                              nodes_fem = data['nodes_fem'],
                                              feature_norms = data['feature_norms'],
                                              num_branch_inputs=data_generator.num_features,
                                              num_basis_functions=num_basis_functions,
                                              permutation_indices = data['permutation_indices_don'])


      trainer = Trainer()
      found_DON = trainer.load_model(deeponet)


      if(args.force_retrain or found_DON==False):
        trainer.fit(deeponet, data, num_of_epochs=args.epochs,
                                    batch_size=args.batch_size,
                                    plot_interval=args.plot_interval,
                                    should_save=True)
    else:
      pass



def test_solver(problem_type, nlevels, solver_params, num_basis_functions):

    # data_generator  = problem_type(mesh_path =" mesh/cylinder_coarse.e", sample_k=True)
    data_generator  = problem_type(mesh_path =" mesh/cylinder_coarse.e", sample_k=False, k_value=6)
    data            = data_generator.get_data(  percentage_test=args.percentage_test,
                                                force_create=args.recreate_data)


    path = 'outputs'
    os.makedirs(path, exist_ok=True)
    model_path = os.path.join('models', args.model_name)

    deeponet = DeepOnet3DNonNestedMeshes(   nodes_don = data['nodes_don'],
                                            nodes_fem = data['nodes_fem'],
                                            feature_norms = data['feature_norms'],
                                            num_branch_inputs=data_generator.num_features,
                                            num_basis_functions=num_basis_functions,
                                            permutation_indices = data['permutation_indices_don'])


    trainer = Trainer()
    trainer.load_model(deeponet)

    csv_out_name = args.solver_output_name


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # solver test # # # # # # # # # # # # # # # # # # # # # # # # #
    num_test_cases  = 10
    num_test_cases  = num_test_cases if len(data['features_fem_test'])-1 < num_test_cases else len(data['features_fem_test'])-1
    sample_ids      = np.random.randint(0, len(data['features_fem_test'][0])-1, num_test_cases)

    its       = np.zeros((num_test_cases))   
    timmings  = np.zeros((num_test_cases))   

    interpolators_list = []
    fe_problems = []
    for l in range(0, nlevels):
      # fe_problems.append(problem_type(mesh_path =" mesh/cylinder_coarse.e", ref_level=l, sample_k=True, linear_solver=None))
      fe_problems.append(problem_type(mesh_path =" mesh/cylinder_coarse.e", ref_level=l, sample_k=False, k_value=6))
      don_transfer = FEMtoDONTranfer(fe_problems[-1].mesh, fe_problems[-1].don_mesh)
      interpolators_list.append(don_transfer)


    for idx, sample_id in enumerate(sample_ids): 
      args.solver_output_name = csv_out_name + "_" + str(idx)

      # append f - fine level 
      features = []
      for f_id in range(len(data['features_fem_test'])):
        sampled_features = data['features_fem_test'][f_id][sample_id]
        result = fe_problems[l].project_features(sampled_features, fe_problems[0].mesh, fe_problems[-1].mesh)
        features.append(np.copy(result))
      
      # finest problem assembly
      F, u, bcs = fe_problems[-1].get_functional_and_bcs(features)


      appctx = {}
      appctx["deepOnet"]      = deeponet
      appctx["interpolators"] = interpolators_list


      # TO SETUP EXAMPLE
      appctx["features"]  = []
      for f_id in range(len(data['features_fem_test'])):
        appctx["features"].append(data['features_don_test'][f_id][sample_id])

      appctx["fe_problems"]=[]
      for l in range(0, nlevels):
        appctx["fe_problems"].append(fe_problems[l])

      with u.dat.vec as sol_petsc:
          sol_petsc.scale(0.0)

      problem = NonlinearVariationalProblem(F, u, bcs=bcs)
      solver  = NonlinearVariationalSolver(problem, solver_parameters=solver_params, appctx=appctx)


      # solver.snes.getKSP().cancelMonitor()
      solver.snes.getKSP().setMonitor(MyKSPMonitorSaveCSV)


      OptDB = PETSc.Options()
      prefix = solver.snes.getKSP().getOptionsPrefix()
      OptDB.setValue(prefix + 'ksp_norm_type', "unpreconditioned")      
      OptDB.setValue(prefix + 'pc_mg_multiplicative_cycles', 1)
      
      solver.snes.getKSP().setFromOptions()

      start_time = time.time()
      
      with u.dat.vec as sol_petsc:
          sol_petsc.scale(0.0)

      solver.solve()
      
      its[idx] = solver.snes.getKSP().getIterationNumber()
      timmings[idx] = (time.time() - start_time)

      logger.info("Solver time: " + str(timmings[idx])+ " seconds    and " + str(its[idx]) + "  iterations. ")

    average_its = np.average(its)
    average_timmings = np.average(timmings)
    logger.info("Average num. its:   " + str(average_its) + "  timmings  " + str(average_timmings))

    return average_its, average_timmings


if __name__ == '__main__':

    current_dir = os.getcwd()
    sys.pycache_prefix=current_dir+"/.cache/Python"  
    
    num_samples_total   = [args.num_samples_total]
    res_per_samples     = [1]
    num_basis_functions = [args.num_basis_functions]


    args.recreate_data = False

    args.seed=123456
    args.csv_stat_name  = "csv_stat_hints.csv"

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


    problem_type = Helmholtz3D_cylinder

    for exp_id in range(len(num_samples_total)): 
      for num_id in range(len(res_per_samples)): 
        for basis_id in range(len(num_basis_functions)): 

          # doesnt make sense to investigate other options with less than 1 sample
          if(args.sampling_method>1 and res_per_samples[num_id]==1):
            continue

          args.num_samples    = int(num_samples_total[exp_id]/res_per_samples[num_id])
          args.res_per_sample = res_per_samples[num_id]

          args.model_name     = "NonNestedHelm3D_"+str(num_samples_total[exp_id]) +"_"+str(num_basis_functions[basis_id]) + "_" + str(args.dofs_don) + "_" + str(args.k_sigma)



          if(args.k_sigma==6 and args.num_samples_total==10000): 
            
            # load stored dataset
            # preprocess_data_to_fit_DON_size() 
            args.dataset_name = "NonNestedHelm3D_k6_10000_DON_size_8.pkl"
            logger.info("args.dataset_name " + str(args.dataset_name))     
            

          else:
            # create the dataset 
            args.dataset_name   = "NonNestedHelm3D_"+str(args.num_samples_total) + "_" + str(args.dofs_don) + "_" + str(args.k_sigma) + ".pkl"




          train_net(problem_type, num_basis_functions[basis_id])

          if(args.only_train or args.only_generate_data):
            continue


          # TODO:: add options from outside flag 
          solver_params = {"snes_type": "ksponly",
                            "snes_monitor": None,
                            "snes_atol": 1e-8,
                            "snes_rtol": 1e-12,
                            "snes_stol": 1e-12,
                            "ksp_atol": 1e-9,
                            "ksp_stol": 1e-12,
                            "ksp_rtol": 1e-12,
                            "ksp_max_it": 500000,
                            "ksp_monitor": None,
                            "ksp_type": "fgmres", 
                            "pc_type": "mg", 
                            # "mg_levels_1_ksp_type": "richardson", 
                            # "mg_levels_1_pc_type": "jacobi", 
                            # "mg_levels_2_ksp_type": "richardson", 
                            # "mg_levels_2_pc_type": "jacobi", 
                            # "mg_levels_3_ksp_type": "richardson", 
                            # "mg_levels_3_pc_type": "jacobi", 
                            # "mg_levels_4_ksp_type": "richardson", 
                            # "mg_levels_4_pc_type": "jacobi", 
                            # "mg_levels_5_ksp_type": "richardson", 
                            # "mg_levels_5_pc_type": "jacobi", 
                            # "mg_levels_6_ksp_type": "richardson", 
                            # "mg_levels_6_pc_type": "jacobi",                             
                            "mg_levels_1_ksp_type": "richardson", 
                            "mg_levels_1_pc_type": "python", 
                            "mg_levels_1_pc_python_type": "petsc_solvers.JacobiHINTS",  
                            "mg_levels_2_ksp_type": "richardson", 
                            "mg_levels_2_pc_type": "python", 
                            "mg_levels_2_pc_python_type": "petsc_solvers.JacobiHINTS",  
                            "mg_levels_3_ksp_type": "richardson", 
                            "mg_levels_3_pc_type": "python", 
                            "mg_levels_3_pc_python_type": "petsc_solvers.JacobiHINTS",  
                            "mg_levels_4_ksp_type": "richardson", 
                            "mg_levels_4_pc_type": "python", 
                            "mg_levels_4_pc_python_type": "petsc_solvers.JacobiHINTS",  
                            "mg_levels_5_ksp_type": "richardson", 
                            "mg_levels_5_pc_type": "python", 
                            "mg_levels_5_pc_python_type": "petsc_solvers.JacobiHINTS",
                            "mg_levels_6_ksp_type": "richardson", 
                            "mg_levels_6_pc_type": "python", 
                            "mg_levels_6_pc_python_type": "petsc_solvers.JacobiHINTS",                            
                            }    
          

          for levels in range(3, 6):
            args.solver_output_name = "outputs/NonNestedHelm3D_example_"+str(args.num_samples) + "_"+ str(args.res_per_sample)+ "_" + str(args.dofs_don) + "_"+ str(levels) + "_" + str(args.sampling_method) + '_' + str(num_basis_functions[basis_id])
            average_its, average_time = test_solver(problem_type, levels, solver_params, num_basis_functions[basis_id])

            df = pd.DataFrame({'sampling_method':[args.sampling_method], 'num_samples_total':[args.num_samples*args.res_per_sample], 'lr':[args.lr], 'num_samples_unique':[args.num_samples], 'res_per_sample':[args.res_per_sample], 'num_basis_functions':[num_basis_functions[basis_id]], 'level':[levels], 'its':[average_its], 'time':[average_time]})
            if not os.path.isfile(args.csv_stat_name):
                df.to_csv(args.csv_stat_name, index=False,  header=True, mode='a')      
            else:
                df.to_csv(args.csv_stat_name, index=False, header=False, mode='a')      
          

