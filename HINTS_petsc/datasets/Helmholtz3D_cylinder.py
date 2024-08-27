try:
    import sys, petsc4py
    petsc4py.init(sys.argv)
    from petsc4py import PETSc
    from pyop2.mpi import COMM_WORLD
except ImportError:
  print("Petsc not found")     
  
import os
import copy
import numpy as np

try:
    from firedrake import MeshHierarchy, Mesh, File,  BoxMesh, FunctionSpace, DirichletBC, as_tensor, Constant, SpatialCoordinate, And, Or
    from firedrake import TrialFunction, TestFunction, Function, assemble, inner, dx, grad, pi, sin, conditional
    from firedrake import NonlinearVariationalSolver, NonlinearVariationalProblem, interpolate,as_ufl
except ImportError:
  print("Firedrake not found")   

from datasets.LinearProblemFiredrakeSampler import LinearProblemFiredrakeSampler, FEMtoDONTranfer

from config import params as args
from utils import logger 


class Helmholtz3D_cylinder(LinearProblemFiredrakeSampler):
                                                                                                 # 8, 16, 24, 32, 64
    def __init__(self, mesh_path, k_value=3, ref_level = 0, linear_solver=None, sample_k=False, ncells=args.dofs_don):
        super(Helmholtz3D_cylinder, self).__init__(linear_solver=linear_solver, dim=2)
        
        self.mesh = Mesh(mesh_path)


        if(ref_level > 0):
            hierarchy = MeshHierarchy(self.mesh, ref_level)
            self.mesh = hierarchy[-1]


        self.don_mesh = BoxMesh(ncells, ncells, ncells, 2.2, 2.2, 1.0, comm=COMM_WORLD)
        self.don_mesh.coordinates.dat.data_wo[:, 0] +=-1.1
        self.don_mesh.coordinates.dat.data_wo[:, 1] +=-1.1        


        self.V_fem = FunctionSpace(self.mesh ,"CG", 1)
        self.V_don = FunctionSpace(self.don_mesh ,"CG", 1)


        self.nodes      = self.get_vertices(self.mesh, self.V_fem)
        self.nodes_don  = self.get_vertices(self.don_mesh, self.V_don)
        logger.info("Number of nodes DON: "+str(len(self.nodes_don[:,0])))
        logger.info("Number of nodes FEM: "+str(len(self.nodes[:,0])))

        self.permutation_indices_don = np.lexsort((self.nodes_don[:,0], self.nodes_don[:,1], self.nodes_don[:,2]))

        self.bc         = [DirichletBC(self.V_fem, 0.0, [1])] # Boundary condition

        self.u_trial    = TrialFunction(self.V_fem)
        self.v_test     = TestFunction(self.V_fem)
        self.f_func     = Function(self.V_fem)

        if(sample_k==False):
            self.k_fixed    = k_value
            self.sample_k   = False
        else:
            self.k_fixed        = args.k_sigma
            self.sample_k       = True
            self.k_func         = Function(self.V_fem)
            self.num_features   = 2

        with self.f_func.dat.vec as vv:
            coef_petsc = vv

        self.ownership_range = coef_petsc.getOwnershipRange()

        m = inner(self.u_trial, self.v_test)*dx
        self.M = assemble(m)



    def assemble_sample(self, sampled_features):

        F, sol, bc = self.get_functional_and_bcs(sampled_features)

        solver_params = {"snes_type": "newtonls",
                          # "snes_monitor": None,
                          "snes_atol": 1e-12,
                          "snes_rtol": 1e-12,
                          "snes_stol": 1e-12,
                          "ksp_atol": 1e-13,
                          "ksp_stol": 1e-13,
                          "ksp_rtol": 1e-13,
                          "ksp_type": "gmres",
                          "pc_type": "lu",
                          "ksp_max_it": 100000,
                          "snes_max_it": 10000}


        problem = NonlinearVariationalProblem(F, sol, bcs=bc)
        solver  = NonlinearVariationalSolver(problem, solver_parameters=solver_params)

        with sol.dat.vec as sol_petsc:
            sol_petsc.set(0.0)
                    
        solver.solve()
        return sol


    def get_functional_and_bcs(self, sampled_features):

        self.sol                = Function(self.V_fem)
        self.f_func.dat.data[:] = sampled_features[0][self.ownership_range[0]:self.ownership_range[1]]            

        if(self.sample_k == False):
            F = inner(grad(self.sol), grad(self.v_test))*dx - self.k_fixed**2 * inner(self.sol, self.v_test)*dx - self.f_func*self.v_test*dx
        else:
            self.k_func.dat.data[:] = sampled_features[1][self.ownership_range[0]:self.ownership_range[1]]            
            F = inner(grad(self.sol), grad(self.v_test))*dx - self.k_func**2 * inner(self.sol, self.v_test)*dx - self.f_func*self.v_test*dx

        return F, self.sol, self.bc



    def sample_rhs(self):
        # generate ks
        sampled_rhs = self.generate_grf_function_3D(    sigma_0=(args.f_sigma,),
                                                        l_0=(args.f_lo,),
                                                        num_of_samples=args.num_samples)

        return sampled_rhs


    def sample_k_coef(self):
        # generate ks
        sampled_k = self.generate_grf_function_3D(      sigma_0=(args.k_sigma,),
                                                        l_0=(args.k_lo,),
                                                        num_of_samples=args.num_samples)
        return sampled_k


    def sample(self):
        np.random.seed(args.seed)
        u_all   = np.zeros((args.num_samples * args.res_per_sample, self.ownership_range[1]-self.ownership_range[0]))   
        
        # generate rhs
        self.sampled_features_fem =[self.sample_rhs()]

        if(self.sample_k): 
            self.sampled_features_fem.append(self.sample_k_coef())


        # np.set_printoptions(threshold=sys.maxsize)
        self.sampled_features_don = [np.zeros((self.sampled_features_fem[0].shape[0],  len(self.nodes_don[:,0])))]
        for f_id  in range(1, len(self.sampled_features_fem)):
            self.sampled_features_don.append(np.zeros((self.sampled_features_fem[f_id].shape[0],  len(self.nodes_don[:,0]))))


        logger.info("Creating DON features")
        don_to_fem_transfer = FEMtoDONTranfer(self.mesh, self.don_mesh)

        for f_id  in range(1, len(self.sampled_features_fem)):
            for i in range(self.sampled_features_fem[f_id].shape[0]):
                don_to_fem_transfer.k_func.dat.data[:] = self.sampled_features_fem[f_id][i, :]
                don_to_fem_transfer.k_func_target = assemble(don_to_fem_transfer.interpolate())
                self.sampled_features_don[f_id][i, :] = don_to_fem_transfer.k_func_target.dat.data[:]
                if(i%10==0):
                    logger.info("Data generation (DON features), case index: "+str(i))            


        logger.info("Creating Solutions ")
        for case_index in range(0,len(self.sampled_features_fem[0])):
            if(self.sample_k==False):
                sol = self.assemble_sample([self.sampled_features_fem[case_index]])
            else:
                sol = self.assemble_sample([self.sampled_features_fem[0][case_index], self.sampled_features_fem[1][case_index]])

            with sol.dat.vec as vv_sol:
                sol_petsc = vv_sol

            u_all[case_index] = sol_petsc.getArray()

            if(case_index%10==0):
                logger.info("Data generation (solutions), case index: "+str(case_index))
        

        self.sampled_features_fem = self.sampled_features_fem
        self.sampled_features_don = self.sampled_features_don



        return u_all


    def create_data(self):
        if(self.mesh.comm.Get_size()>1):
            logger.info('Generation of the data in parallel is not supported yet (some operations are missing). Rerun is serial.')
            exit(0)

        u_fem = self.sample()

        data_preprocessed = {   'nodes_fem': self.nodes,
                                'nodes_don': self.nodes_don,
                                'permutation_indices_don': self.permutation_indices_don,
                                'features_fem': self.sampled_features_fem,
                                'features_don': self.sampled_features_don,
                                'outputs_fem': u_fem }

        return data_preprocessed





    def prepare_data(self, data_preprocessed, percent_test, should_save=True):
        features_fem    = data_preprocessed['features_fem']
        features_don    = data_preprocessed['features_don']

        outputs_fem     = data_preprocessed['outputs_fem']

        logger.debug('Data shape:')
        logger.debug('     Output u shape: ' + str(outputs_fem.shape))

        num_of_samples = outputs_fem.shape[0]
        
        # split to test and train set
        num_of_test_samples = int(round(num_of_samples * percent_test))
        num_of_train_samples = num_of_samples - num_of_test_samples

        output_train  = outputs_fem[:num_of_train_samples]
        output_test   = outputs_fem[num_of_train_samples:]

        # features
        features_fem_train  = []
        features_don_train  = []
        features_fem_test   = []
        features_don_test   = []        


        for i in range(0, len(features_fem)):
            features_fem_train.append(features_fem[i][:num_of_train_samples])
            features_don_train.append(features_don[i][:num_of_train_samples])

            features_fem_test.append(features_fem[i][num_of_train_samples:])
            features_don_test.append(features_don[i][num_of_train_samples:])

        feature_norms = []

        logger.info("features_don_train " + str(features_don_train[0].shape))
        logger.info("features_fem_train " + str(features_fem_train[0].shape))

        # norms = [1.0, 1.0]
        # feature_norms.append(norms)
        
        logger.debug('Train-test ratios:')
        logger.debug('     Shape of f train samples: ' + str(features_fem_train[0].shape))
        logger.debug('     Shape of output train samples: ' + str(output_train.shape))
        logger.debug('     Shape of f test samples: ' + str(features_fem_test[0].shape))
        logger.debug('     Shape of output test samples: ' + str(output_test.shape))


        data_processed = {  'nodes_fem': data_preprocessed.get('nodes_fem'),
                            'nodes_don': data_preprocessed.get('nodes_don'),
                            'permutation_indices_don': self.permutation_indices_don,
                            'feature_norms': feature_norms,
                            'permutation_indices_don': data_preprocessed.get('permutation_indices_don'),
                            'u_train': output_train,
                            'u_test': output_test,
                            'features_fem_train': features_fem_train,
                            'features_fem_test': features_fem_test,
                            'features_don_train': features_don_train,
                            'features_don_test': features_don_test}

        if should_save:
            self.save(data_processed)

        return data_processed
