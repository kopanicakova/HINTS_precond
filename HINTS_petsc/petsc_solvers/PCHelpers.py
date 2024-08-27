import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

import os
import numpy as np
from scipy.sparse import csr_matrix
from scipy.interpolate import interp2d, interpn
from torch import load as torch_load
import pandas as pd
from config import params as args
import time


__all__ = ("MyKSPMonitorSaveCSV", "MYKSPConvergenceTest", "IdentityPC", "MyKSP")



def MYKSPConvergenceTest(ksp, its, rnorm):
    rtol, atol, divtol, max_its = ksp.getTolerances()
    a_norm                      = ksp.getResidualNorm()

    if(its >= max_its):
        return True
    elif(a_norm <= atol):
        return True    
    elif(rnorm <= rtol):
        return True    
    else:
        return False




def MyKSPMonitorSaveCSV(ksp, it, norm):
    # if(it==0):
    #     print("It     ", "     || r ||   ")    
    #     print("--------------------------")    

    # print(it,"    ", norm)

    A,P             = ksp.getOperators()
    problem_size    = A.getSize()[0]
    ksp_type        = ksp.getType()

    if(it==0):
        args.res_norm0  = norm
        args.time0      = time.time() # not super accurate, but could be way worse


    # if(ksp_type=="python"):
    #     ksp_python = ksp.getPythonContext()

    df = pd.DataFrame({ 'it':[it], 'r_norm':[norm], 'rel_norm':[norm/args.res_norm0], 'time': [time.time() - args.time0]})
    if not os.path.isfile(args.solver_output_name):
        df.to_csv(args.solver_output_name, index=False,  header=True, mode='a')      
    else:
        df.to_csv(args.solver_output_name, index=False, header=False, mode='a')     


    # else:
    # # output_name = 'outputs/ksp_log_' + str(problem_size) + 'dofs_' +str(ksp.getType()) + '_pc_'+ str(ksp.getPC().getType()) + '.csv'
    #     df = pd.DataFrame({ 'it':[it], 'r_norm':[norm]})
    #     if not os.path.isfile(args.solver_output_name):
    #         df.to_csv(args.solver_output_name, index=False,  header=True, mode='a')      
    #     else:
    #         df.to_csv(args.solver_output_name, index=False, header=False, mode='a')      




class BasePC(object):
    # def __init__(self, non_bc=None):
        # self.non_bc = non_bc
    # def __init__(self):
    #   pass

    def setup(self, pc):
        pass
    def reset(self, pc):
        pass
    def apply(self, pc, x, y):
        raise NotImplementedError
    def applyT(self, pc, x, y):
        self.apply(pc, x, y)
    def applyS(self, pc, x, y):
        self.apply(pc, x, y)
    def applySL(self, pc, x, y):
        self.applyS(pc, x, y)
    def applySR(self, pc, x, y):
        self.applyS(pc, x, y)
    def applyRich(self, pc, x, y, w, tols):
        self.apply(pc, x, y)


class IdentityPC(BasePC):
    def __init__(self):
        self.residuals  = []
        pass    

    def apply(self, pc, x, y):
        # print("---")
        x.copy(y)
        self.residuals.append(x.getArray().copy())


class MyKSP(object):

    def __init__(self):
        self.verbose    =   True
        self.start_time =   0.0
        self.end_time   =   0.0
        self.r0_norm    =   0.0
        pass

    def create(self, ksp):
        self.work = []

    def destroy(self, ksp):
        for v in self.work:
            v.destroy()

    def setUp(self, ksp):
        self.work[:] = ksp.getWorkVecs(right=2, left=None)

    def reset(self, ksp):
        for v in self.work:
            v.destroy()
        del self.work[:]

    def get_execution_time(self):
        return self.end_time - self.start_time

    def loop(self, ksp, r):
        its = ksp.getIterationNumber()
        rnorm = r.norm()

        if(its==0):
            self.start_time = time.time()
            self.r0_norm    = rnorm
        else:
            self.end_time   = time.time()

        ksp.setResidualNorm(rnorm)
        ksp.logConvergenceHistory(rnorm)
        ksp.monitor(its, rnorm)
        reason = ksp.callConvergenceTest(its, rnorm)


        if(self.verbose):
            print(its, "     ", rnorm)

        if(its+1 > self.max_its ): 
            ksp.setConvergedReason(1)
            return 1
        elif not reason:
            ksp.setIterationNumber(its+1)
        else:
            ksp.setConvergedReason(reason)
        return reason


    def loop_norm(self, ksp, rnorm):
        its = ksp.getIterationNumber()
        ksp.setResidualNorm(rnorm)
        ksp.logConvergenceHistory(rnorm)
        ksp.monitor(its, rnorm)
        reason = ksp.callConvergenceTest(its, rnorm)

        if(self.verbose):
            print(its, "     ", rnorm)

        if(its+1 > self.max_its ): 
            ksp.setConvergedReason(1)
            return 1
        elif not reason:
            ksp.setIterationNumber(its+1)
        else:
            ksp.setConvergedReason(reason)
        return reason



