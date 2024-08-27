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

from deeponets.DeepOnet3DNonNestedMeshes    import DeepOnet3DNonNestedMeshes
from datasets.Helmholtz3D_cylinder   import Helmholtz3D_cylinder
from trainers.Trainer                       import Trainer

import matplotlib.pyplot as plt
from utils import logger
import pandas as pd
from config import params as args

try:
  from firedrake import UnitIntervalMesh, Function, MeshHierarchy, COMM_WORLD
  from firedrake import prolong, NonlinearVariationalSolver, File, NonlinearVariationalProblem
  from petsc_solvers.PCHelpers import * 
except ImportError:
  print("Firedrake/petsc not found")


import warnings
warnings.filterwarnings("ignore")  



def test_solver(problem_type, nlevels, solver_params):

    # data_generator  = problem_type(mesh_path =" mesh/cylinder_coarse.e", sample_k=True)
    data_generator  = problem_type(mesh_path =" mesh/cylinder_coarse.e", sample_k=False, k_value=6)
    data            = data_generator.get_data(  percentage_test=args.percentage_test,
                                                force_create=args.recreate_data)

    csv_out_name = args.solver_output_name
    

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # solver test # # # # # # # # # # # # # # # # # # # # # # # # #
    num_test_cases  = 10
    num_test_cases  = num_test_cases if len(data['features_fem_test'])-1 < num_test_cases else len(data['features_fem_test'])-1
    sample_ids      = np.random.randint(0, len(data['features_fem_test'][0])-1, num_test_cases)

    its       = np.zeros((num_test_cases))   
    timmings  = np.zeros((num_test_cases))   

    fe_problems = []
    for l in range(0, nlevels):
      # fe_problems.append(problem_type(mesh_path =" mesh/cylinder_coarse.e", ref_level=l, sample_k=True, linear_solver=None))
      fe_problems.append(problem_type(mesh_path =" mesh/cylinder_coarse.e", ref_level=l, sample_k=False, k_value=6, linear_solver=None))


    for idx, sample_id in enumerate(sample_ids): 
      args.solver_output_name = csv_out_name + "_" + str(idx)

      # append features - fine level 
      features = []
      for f_id in range(len(data['features_fem_test'])):
        sampled_features = data['features_fem_test'][f_id][sample_id]
        result = fe_problems[l].project_features(sampled_features, fe_problems[0].mesh, fe_problems[-1].mesh)
        features.append(np.copy(result))


      # finest problem assembly
      F, u, bcs = fe_problems[-1].get_functional_and_bcs(features)

      problem = NonlinearVariationalProblem(F, u, bcs=bcs)
      solver  = NonlinearVariationalSolver(problem, solver_parameters=solver_params)

      solver.snes.getKSP().setMonitor(MyKSPMonitorSaveCSV)

      

      OptDB = PETSc.Options()      
      prefix = solver.snes.getKSP().getOptionsPrefix()
      OptDB.setValue(prefix + 'pc_type', "hypre")
      OptDB.setValue(prefix + 'ksp_norm_type', "unpreconditioned")
      OptDB.setValue(prefix + 'pc_hypre_type', "boomeramg")
      OptDB.setValue(prefix + 'pc_hypre_boomeramg_max_iter', 1)              

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
    
    args.seed=12345
    args.csv_stat_name  = "csv_stat_hypre.csv"

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


    problem_type = Helmholtz3D_cylinder


    if(args.k_sigma==6 and args.num_samples_total==10000): 
      # load stored dataset 
      args.dataset_name = "NonNestedHelm3D_k6_10000_DON_size_8.pkl"
      logger.info("args.dataset_name " + str(args.dataset_name))     


    else:
      # create the dataset 
      args.dataset_name   = "NonNestedHelm3D_"+str(args.num_samples_total) + "_" + str(args.dofs_don) + "_" + str(args.k_sigma) + ".pkl"



    # TODO:: config. from outside
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
                      }    


    for levels in range(3, 6):
      args.solver_output_name = "outputs/NonNestedHelm3D_example_hypre"+str(args.num_samples_total) + "_"+ str(levels)
      average_its, average_time = test_solver(problem_type, levels, solver_params)

      df = pd.DataFrame({'level':[levels], 'its':[average_its], 'time':[average_time]})
      if not os.path.isfile(args.csv_stat_name):
          df.to_csv(args.csv_stat_name, index=False,  header=True, mode='a')      
      else:
          df.to_csv(args.csv_stat_name, index=False, header=False, mode='a')      
    
    