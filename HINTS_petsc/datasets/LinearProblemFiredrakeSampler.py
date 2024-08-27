try:
  import sys, petsc4py
  petsc4py.init(sys.argv)
  from petsc4py import PETSc
except ImportError:
  print("Petsc4py not found")

import os
import numpy as np
import matplotlib.pyplot as plt
from abc import  abstractmethod
import utils as doputils

try:
  from pyop2.mpi import COMM_WORLD
  from firedrake import VectorFunctionSpace, Interpolator,  interpolate, assemble, FunctionSpace, Function, interpolate
  from firedrake import NonlinearVariationalSolver, NonlinearVariationalProblem, File
except ImportError:
  print("Firedrake not found")


from config import params as args
from utils import logger, create_folder
import pickle


class FEMtoDONTranfer(object):
  def __init__(self, mesh_base, mesh_target ):
    self.V_base         = FunctionSpace(mesh_base, "CG", 1)
    self.V_target       = FunctionSpace(mesh_target, "CG", 1)

    self.k_func         = Function(self.V_base)
    self.interpolator   = Interpolator(self.k_func, self.V_target, allow_missing_dofs=True)

    # to allocate
    assemble(self.interpolate())

  def interpolate(self): 
    return self.interpolator.interpolate()




class LinearProblemFiredrakeSampler(object):
    def __init__(self, linear_solver, dim):
        self.dim                        =   dim
        self.linear_solver              =   linear_solver
        self.sampled_features           =   None
        self.nodes                      =   None
        self.permutation_indices        =   None
        self.inv_permutation_indices    =   None

        self.num_features               =   1


    @abstractmethod
    def assemble_sample(self, sampled_features):
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        raise NotImplementedError

    @abstractmethod
    def enforceBC(self):
        raise NotImplementedError

    @abstractmethod
    def get_functional_and_bcs(self, sampled_features):
        raise NotImplementedError

    @abstractmethod
    def get_rhs_function(self):
        raise NotImplementedError


    def get_vertices(self, mesh, FunctionSpace):
        # interpolate the coordinates onto the nodes of W.
        W = VectorFunctionSpace(mesh, FunctionSpace.ufl_element())
        X = assemble(interpolate(mesh.coordinates, W))

        return X.dat.data_ro


    def project_features(self, base_feature, base_mesh,  target_mesh):
        
        # TODO:: fix spaces
        V_base      = FunctionSpace(base_mesh, "CG", 1)
        V_target    = FunctionSpace(target_mesh, "CG", 1)

        k_func      = Function(V_base)

        k_func.dat.data[:]  = base_feature[:]
        k_func_target       = assemble(interpolate(k_func, V_target, allow_missing_dofs=True))


        with k_func_target.dat.vec as vv:
            projected_features_numpy = vv.getArray()

        return projected_features_numpy
        


    def project_features_to_don(self, base_feature, base_mesh):
        return self.project_features(base_feature, base_mesh, self.don_mesh)
        


    def sample_residuals(self):

        if(args.sampling_method<=4):
            residuals   = self.generate_grf_function(   sigma_0=(args.res_sigma,),
                                                        l_0=(args.res_lo,),
                                                        num_of_samples=args.num_samples * args.res_per_sample)             


        # Lets just keep GRFS
        if(args.sampling_method==1):
            pass
        # Modify GRFS to keep rhs and zeros
        elif(args.sampling_method==2):
            # For each sample add zero residual
            zero_res        = residuals[::args.res_per_sample]
            zero_res[:]     = 0.0
            # print("residuals ", residuals.shape)

            # For each sample add actual RHS 
            rhs_fun         = self.get_rhs_function()
            zero_res        = residuals[1::args.res_per_sample]
            zero_res[:]     = rhs_fun.dat.data[:]   
        
        elif(args.sampling_method==3):
            # For each sample add actual RHS 
            rhs_fun     = self.get_rhs_function()
            zero_res    = residuals[1::args.res_per_sample]
            zero_res[:] = rhs_fun.dat.data[:]        

        elif(args.sampling_method==4):
            # For each sample add zero residual
            zero_res        = residuals[::args.res_per_sample]
            zero_res[:]     = 0.0
        
        elif(args.sampling_method==5):
            min_sampling_rate = 1./args.dofs_don
            max_sampling_rate = 0.5

            los = np.linspace(min_sampling_rate, max_sampling_rate, num=args.res_per_sample, endpoint=True)

            residuals = None

            # TODO:: investigate role of sigma
            for l in los:
                resids = self.generate_grf_function(    sigma_0=(args.res_sigma,),
                                                        l_0=(l,),
                                                        num_of_samples=args.num_samples * 1)                             

                if(residuals is not None): 
                    residuals = np.vstack((residuals, resids))
                else:
                    residuals = resids
        
        # going down by factor of 2
        elif(args.sampling_method==6):
            min_sampling_rate = 1./args.dofs_don
            max_sampling_rate = 0.5

            # print("min ", min_sampling_rate, "   max: ", max_sampling_rate)
            # exit(0)

            los = []

            rate = max_sampling_rate
            for idx in range(args.res_per_sample):
                los.append(rate)
                rate = rate/2


            residuals = None

            # TODO:: investigate role of sigma
            for l in los:
                resids = self.generate_grf_function(    sigma_0=(args.res_sigma,),
                                                        l_0=(l,),
                                                        num_of_samples=args.num_samples * 1)                             

                if(residuals is not None): 
                    residuals = np.vstack((residuals, resids))
                else:
                    residuals = resids

        # sampling random residuals 
        elif(args.sampling_method==7):

            residuals = None            
            random_rhs = np.random.rand(self.sampled_features[0].shape[0], self.nodes.shape[0])

            for case_index in range(0,len(self.sampled_features[0]), args.res_per_sample):
                F, sol, bc = self.get_functional_and_bcs([self.sampled_features[0][case_index], random_rhs[case_index]])

                solver_params = {"snes_type": "newtonls",
                                  # "snes_monitor": None,
                                  # "ksp_monitor": None,
                                  "snes_atol": 1e-13,
                                  "snes_rtol": 1e-13,
                                  "snes_stol": 1e-13,
                                  "ksp_atol": 1e-12,
                                  "ksp_stol": 1e-12,
                                  "ksp_rtol": 1e-12,
                                  "ksp_type": "gmres",
                                  "pc_type": "python",
                                  "pc_python_type": "petsc_solvers.IdentityPC", 
                                  "ksp_max_it":  100000,
                                  "snes_max_it": 10000}

                with sol.dat.vec as sol_petsc:
                    sol_np = sol_petsc.getArray()
                    sol_np[:] = (np.random.rand(sol_np.shape[0])<.5)*2 - 1
                    # sol_np[:] = np.ones_like(sol_np)

                problem = NonlinearVariationalProblem(F, sol, bcs=bc)
                solver  = NonlinearVariationalSolver(problem, solver_parameters=solver_params)

                solver.solve() 

                pc          = solver.snes.getKSP().getPC().getPythonContext()
                residualsss = np.stack( pc.residuals, axis=0)
                idxs        = np.random.randint(0, high=residualsss.shape[0], size=args.res_per_sample)
                resids      = residualsss[idxs, :]

                if(residuals is not None): 
                    residuals = np.vstack((residuals, resids))
                else:
                    residuals = resids
            

        # sampling smoothed vectors 
        elif(args.sampling_method==8):

            residuals = None
            random_rhs = np.random.rand(self.sampled_features[0].shape[0], self.nodes.shape[0])

            for case_index in range(0,len(self.sampled_features[0])):
                F, sol, bc = self.get_functional_and_bcs([self.sampled_features[0][case_index], random_rhs[case_index]])

                solver_params = {"snes_type": "newtonls",
                                  # "snes_monitor": None,
                                  # "ksp_monitor": None,
                                  "snes_atol": 1e-13,
                                  "snes_rtol": 1e-13,
                                  "snes_stol": 1e-13,
                                  "ksp_atol": 1e-12,
                                  "ksp_stol": 1e-12,
                                  "ksp_rtol": 1e-12,
                                  "ksp_type": "gmres",
                                  "pc_type": "composite",
                                  "pc_composite_type": "multiplicative",
                                  "pc_composite_pcs": "sor,sor,python",
                                  "sub_2_pc_python_type": "petsc_solvers.IdentityPC", 
                                  "ksp_max_it":  100000,
                                  "snes_max_it": 10000}

                with sol.dat.vec as sol_petsc:
                    sol_np = sol_petsc.getArray()
                    sol_np[:] = (np.random.rand(sol_np.shape[0])<.5)*2 - 1


                problem = NonlinearVariationalProblem(F, sol, bcs=bc)
                solver  = NonlinearVariationalSolver(problem, solver_parameters=solver_params)

                solver.solve() 

                pc      = solver.snes.getKSP().getPC().getCompositePC(2).getPythonContext()
                resids  =  pc.residuals[0]
                resids  = np.reshape(resids, (1, self.nodes.shape[0]))


                if(residuals is not None): 
                    residuals = np.vstack((residuals, resids))
                else:
                    residuals = resids

        # Eigen-decomp
        elif(args.sampling_method==9):

            residuals = None

            random_rhs = np.random.rand(self.sampled_features[0].shape[0], self.nodes.shape[0])

            for case_index in range(0,len(self.sampled_features[0]), args.res_per_sample):
                F, sol, bc = self.get_functional_and_bcs([self.sampled_features[0][case_index], random_rhs[case_index]])

                resids_collection = None
                for num_solves in range(args.res_per_sample):
                    solver_params = {"snes_type": "newtonls",
                                          # "snes_monitor": None,
                                          # "ksp_monitor": None,
                                          "snes_atol": 1e-13,
                                          "snes_rtol": 1e-13,
                                          "snes_stol": 1e-13,
                                          "ksp_atol": 1e-12,
                                          "ksp_stol": 1e-12,
                                          "ksp_rtol": 1e-12,
                                          "ksp_type": "gmres",
                                          "pc_type": "composite",
                                          "pc_composite_type": "multiplicative",
                                          "pc_composite_pcs": "sor,sor,python",
                                          "sub_2_pc_python_type": "petsc_solvers.IdentityPC", 
                                          "ksp_max_it":  100000,
                                          "snes_max_it": 10000}

                    with sol.dat.vec as sol_petsc:
                        sol_np = sol_petsc.getArray()
                        sol_np[:] = (np.random.rand(sol_np.shape[0])<.5)*2 - 1


                    problem = NonlinearVariationalProblem(F, sol, bcs=bc)
                    solver  = NonlinearVariationalSolver(problem, solver_parameters=solver_params)

                    solver.solve() 

                    pc      = solver.snes.getKSP().getPC().getCompositePC(2).getPythonContext()
                    resids  = pc.residuals[0]
                    resids  = np.reshape(resids, (1, self.nodes.shape[0]))


                    if(resids_collection is not None): 
                        resids_collection = np.vstack((resids_collection, resids))
                    else:
                        resids_collection = resids    


                # resids_collection
                U, S, Vh = np.linalg.svd(resids_collection.T, full_matrices=True)
                resids_collection = U[0:args.res_per_sample,:]


                if(residuals is not None): 
                    residuals = np.vstack((residuals, resids_collection))
                else:
                    residuals = resids_collection                    




        # Eigen-decomp 2
        elif(args.sampling_method==10):

            residuals = None

            random_rhs = np.random.rand(self.sampled_features[0].shape[0], self.nodes.shape[0])

            for case_index in range(0,len(self.sampled_features[0]), args.res_per_sample):
                F, sol, bc = self.get_functional_and_bcs([self.sampled_features[0][case_index], random_rhs[case_index]])

                resids_collection = None
                for num_solves in range(args.res_per_sample):
                    solver_params = {"snes_type": "newtonls",
                                          # "snes_monitor": None,
                                          # "ksp_monitor": None,
                                          "snes_atol": 1e-13,
                                          "snes_rtol": 1e-13,
                                          "snes_stol": 1e-13,
                                          "ksp_atol": 1e-12,
                                          "ksp_stol": 1e-12,
                                          "ksp_rtol": 1e-12,
                                          "ksp_type": "gmres",
                                          "pc_type": "composite",
                                          "pc_composite_type": "multiplicative",
                                          "pc_composite_pcs": "sor,sor,python",
                                          "sub_2_pc_python_type": "petsc_solvers.IdentityPC", 
                                          "ksp_max_it":  100000,
                                          "snes_max_it": 10000}

                    with sol.dat.vec as sol_petsc:
                        sol_np = sol_petsc.getArray()
                        sol_np[:] = (np.random.rand(sol_np.shape[0])<.5)*2 - 1


                    problem = NonlinearVariationalProblem(F, sol, bcs=bc)
                    solver  = NonlinearVariationalSolver(problem, solver_parameters=solver_params)

                    solver.solve() 

                    pc      = solver.snes.getKSP().getPC().getCompositePC(2).getPythonContext()
                    resids  = pc.residuals[0]
                    resids  = np.reshape(resids, (1, self.nodes.shape[0]))


                    if(resids_collection is not None): 
                        resids_collection = np.vstack((resids_collection, resids))
                    else:
                        resids_collection = resids    


                # resids_collection
                U, S, Vh = np.linalg.svd(resids_collection.T, full_matrices=True)
                resids_collection = U[:,0:args.res_per_sample]


                if(residuals is not None): 
                    residuals = np.vstack((residuals, resids_collection.T))
                else:
                    residuals = resids_collection.T    


        self.sampled_features.append(residuals)




    def generate_grf_function(self, sigma_0, l_0, num_of_samples):
        if(self.dim==1): 
            return self.generate_grf_function_1D(sigma_0, l_0, num_of_samples)
        elif(self.dim==2):
            return self.generate_grf_function_2D(sigma_0, l_0, num_of_samples)
        else:
            logger.info('The 3D GRF generation is missing at the moment:   ')
            exit(0)


    # TODO:: make work in parallel with Petsc parallel distribution
    def generate_grf_function_1D(self, sigma_0, l_0, num_of_samples):

        num_of_elements = args.dofs_don-1

        if len(sigma_0) != len(l_0):
            raise ValueError('Size of sigma_0 and l_0 mismatch')
        x_grid = np.linspace(0, 1, num_of_elements + 1)
        x_distances = x_grid.transpose() - np.expand_dims(x_grid, -1)
        covariance_matrix = np.zeros((num_of_elements + 1, num_of_elements + 1))
        for mode_index, corr_length in enumerate(l_0):
            covariance_matrix = \
                covariance_matrix + ((sigma_0[mode_index] ** 2) *
                                     np.exp(- 0.5 / (corr_length ** 2) *
                                                   (x_distances ** 2)))
        mu = np.zeros_like(x_grid)
        return np.random.multivariate_normal(mu, covariance_matrix, num_of_samples)


    # TODO:: make work in parallel with Petsc parallel distribution
    def generate_grf_function_2D(self, sigma_0, l_0, num_of_samples):
        if len(sigma_0) != len(l_0):
            raise ValueError('Size of sigma_0 and l_0 mismatch')

        x = self.nodes[:,0]
        y = self.nodes[:,1]

        num_of_nodes = self.nodes.shape[0]
        x1, x2 = np.meshgrid(x, x, indexing='ij')
        y1, y2 = np.meshgrid(y, y, indexing='ij')

        distances_squared  = ((x1 - x2) ** 2 + (y1 - y2) ** 2)
        covariance_matrix = np.zeros((num_of_nodes, num_of_nodes))

        for mode_index, corr_length in enumerate(l_0):
            covariance_matrix += ((sigma_0[mode_index] ** 2) *
                                   np.exp(- 0.5 / (corr_length ** 2) *
                                          distances_squared))
        mu = np.zeros_like(x)
        samples = np.random.multivariate_normal(mu, covariance_matrix, num_of_samples)

        return samples



    # TODO:: make work in parallel with Petsc parallel distribution
    def generate_grf_function_3D(self, sigma_0, l_0, num_of_samples):
        if len(sigma_0) != len(l_0):
            raise ValueError('Size of sigma_0 and l_0 mismatch')

        x = self.nodes[:,0]
        y = self.nodes[:,1]
        z = self.nodes[:,2]

        num_of_nodes = self.nodes.shape[0]
        x1, x2 = np.meshgrid(x, x, indexing='ij')
        y1, y2 = np.meshgrid(y, y, indexing='ij')
        z1, z2 = np.meshgrid(z, z, indexing='ij')

        distances_squared   = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        covariance_matrix   = np.zeros((num_of_nodes, num_of_nodes))

        for mode_index, corr_length in enumerate(l_0):
            covariance_matrix += ((sigma_0[mode_index] ** 2) *
                                   np.exp(- 0.5 / (corr_length ** 2) *
                                          distances_squared))
        mu      = np.zeros_like(x)
        samples = np.random.multivariate_normal(mu, covariance_matrix, num_of_samples)

        return samples

    


    def create_data(self):
        if(self.mesh.comm.Get_size()>1):
            logger.info('Generation of the data in parallel is not supported yet (some operations are missing). Rerun is serial.')
            exit(0)

        u_all = self.sample()

        data_preprocessed = {   'nodes': self.nodes,
                                'permutation_indices': self.permutation_indices,
                                'features': self.sampled_features,
                                'outputs': u_all }

        return data_preprocessed


    def save(self, data_processed, save_path=None):
        if save_path is None:
            create_folder('data')
            save_path = os.path.join('data', args.dataset_name)
        pickle.dump(data_processed, open(save_path, 'wb'))



    @abstractmethod
    def prepare_data(self, data_preprocessed, percent_test):
        raise NotImplementedError


    def get_data(self, percentage_test, data_folder='data', force_create=False):
        # return doputils.get_data(percentage_test, data_folder, force_create)
        data_path = os.path.join(data_folder, args.dataset_name)
        if os.path.exists(data_path) and not force_create:
            logger.info('Loading data from file')
            return pickle.load(open(data_path, 'rb'))
        else:
            logger.info('Creating data')
            data_preprocessed = self.create_data()
            return self.prepare_data(data_preprocessed, percentage_test)        

        return data        
