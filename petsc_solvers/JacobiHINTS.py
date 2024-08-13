import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from firedrake.preconditioners.base import PCBase
import numpy as np
from firedrake import *
import torch
from config import params as args
from config import DEVICE
import time

__all__ = ("JacobiHINTS",)

class JacobiHINTS(PCBase):
    def initialize(self, pc):

        A, P    = pc.getOperators()

        A, B    = pc.getOperators()
        self.D  = A.getDiagonal()
        self.D  = 1./self.D      
        self.Ap = self.D.duplicate()  
        self.resid = self.D.duplicate()  
        

        appctx                 = self.get_appctx(pc)
        deeponet               = appctx["deepOnet"]
        fe_problems            = appctx["fe_problems"]

        self.level=0
        for l in range(0, len(fe_problems)): 
            # TODO:: fix for parallel execution 
            if(len(fe_problems[l].nodes)==A.getSizes()[0][0]):
                self.level=l        


        # TODO:: same trick can be done if we have 2 branches
        trunk_inputs                = torch.tensor(fe_problems[self.level].nodes, requires_grad=True,  dtype=args.dtype, device=DEVICE)
        self.precom_trunk_outputs   = deeponet.forward_trunk(trunk_inputs)  


        self.smoothing_factor   = 0.6
        self.num_jacobi_its = 1 # 2

        pass

    def update(self, pc):
        pass        


    def project_features(self, base_feature, interpolator):
        
        
        interpolator.k_func.dat.data[:]  = base_feature[:]
        interpolator.k_func_target = assemble(interpolator.interpolate())

        with interpolator.k_func_target.dat.vec as vv:
            projected_features_numpy = vv.getArray()
                            
        return projected_features_numpy        

    def apply(self, pc, rhs, sol):    

        start_time_top = time.time()
        appctx           = self.get_appctx(pc)
        fe_problems      = appctx["fe_problems"]
        features         = appctx["features"]
        deeponet         = appctx["deepOnet"]
        interpolators    = appctx["interpolators"]

        A, B = pc.getOperators()

        # Jacobi steps 
        for i in range(self.num_jacobi_its): 
            A.mult(sol, self.Ap)
            rhs.copy(self.resid)
            self.resid.axpy(-1.0, self.Ap)  


            #  apply Jacobi step
            self.resid.pointwiseMult(self.D, self.resid)
            sol.axpy(self.smoothing_factor, self.resid)
            

        # HINTS step
        A.mult(sol, self.Ap)
        rhs.copy(self.resid)
        self.resid.axpy(-1.0, self.Ap)  

        features[0]    = self.project_features(self.resid.getArray(), interpolators[self.level])
        deeponet_sol   = deeponet.infer(features, fe_problems[self.level], self.precom_trunk_outputs)
        sol.setArray(deeponet.infer(features, fe_problems[self.level], self.precom_trunk_outputs))


        # Jacobi steps 
        for i in range(self.num_jacobi_its): 
            A.mult(sol, self.Ap)
            rhs.copy(self.resid)
            self.resid.axpy(-1.0, self.Ap)  

            #  apply Jacobi step
            self.resid.pointwiseMult(self.D, self.resid)
            sol.axpy(self.smoothing_factor, self.resid)        


    # almost ...
    applyTranspose = apply                


    def destroy(self, pc):
        pass





