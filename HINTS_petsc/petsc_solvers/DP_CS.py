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

__all__ = ("DP_CS", )

class DP_CS(PCBase):
    def initialize(self, pc):

        A, P = pc.getOperators()
    
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


        features[0]    = self.project_features(rhs.getArray(), interpolators[self.level])
        sol.setArray(deeponet.infer(features, fe_problems[self.level], self.precom_trunk_outputs))

    # almost ...
    applyTranspose = apply                


    def destroy(self, pc):
        pass

