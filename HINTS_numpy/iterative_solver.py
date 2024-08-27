import configs
import utils
import numpy as np
from scipy.interpolate import interp2d
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from deeponet import DeepONet
from torch import load as torch_load
import time
import tracemalloc


class NumericalSolver():
    def __init__(self, k_func, f_func, u_init, logger,
                 init_stiff_p=None,
                 init_f_p=None):
        self.solver_method = configs.SOLVER_METHOD
        self.x_nodes = np.linspace(0, 1, k_func.shape[0])
        if configs.DIMENSIONS == 2:
            # Assuming N_x = N_y
            # TODO: make generic. Problem - in MG, we cannot use the num of elements
            # to break the coordinates vector
            self.y_nodes = np.linspace(0, 1, k_func.shape[1])
        elif configs.DIMENSIONS == 3:
            self.y_nodes = np.linspace(0, 1, k_func.shape[1])
            self.z_nodes = np.linspace(0, 1, k_func.shape[2])
        self.k_func = k_func
        self.f_func = f_func
        self.u_init = u_init
        self.logger = logger
        self.stiff_p = None
        self.f_p = None
        self.m_p = None
        self.n_p = None
        self.m_inv_f_p = None
        self.m_inv_n_p = None
        self.u_true = None
        self.g_mat = None
        self.non_bc = None
        self.f_reverse_translator = None

        if configs.PROBLEM == 'poisson':
            if configs.L_SHAPED:
                self.problem = utils.l_shaped_poisson
            else:
                self.problem = utils.poisson
        elif configs.PROBLEM == 'helmholtz':
            if configs.DIMENSIONS == 1:
                self.problem = utils.helmholtz
            elif configs.DIMENSIONS == 2:
                self.problem = utils.helmholtz_2d
            elif configs.DIMENSIONS == 3:
                self.problem = utils.helmholtz_3d

        if init_stiff_p is None and init_f_p is None:
            self.assemble()
        else:
            self.assemble(init_stiff_p, init_f_p)

        self.eigenvalues, self.eigenvectors = \
            self.compute_sorted_eigenvalues_and_eigenvectors(self.g_mat)

        self.u_approx = None
        if configs.L_SHAPED:  # TODO: generalize
            self.u_init = np.zeros_like(self.u_true)
        if configs.SOLVER_METHOD == 'CG':
            self.r_step = self.f_p - self.stiff_p @ np.zeros((self.f_p.shape[0],))
            self.p_step = self.r_step

    def assemble(self, stiff_p=None, f_p=None):
        # If both None: first time assembly; assemble from func
        # If neither is None: first time assembly; assemble from inputs
        # If stiff_p is None but f_p is not None: update f_p and related vars

        init_from_func = stiff_p is None and f_p is None
        init_from_input = not (stiff_p is None or f_p is None)
        update_f = (stiff_p is None) and not (f_p is None)
        invalid = not (init_from_func or init_from_input or update_f)
        if invalid:
            raise ValueError('Inputs stiff_p and f_p are invalid')

        if init_from_func:
            # TODO: make generic
            if configs.DIMENSIONS == 1:
                outputs = self.problem(self.x_nodes,
                                       self.k_func,
                                       self.f_func)
            elif configs.DIMENSIONS == 2:
                outputs = self.problem(self.x_nodes,
                                       self.y_nodes,
                                       self.k_func,
                                       self.f_func)
            elif configs.DIMENSIONS == 3:
                outputs = self.problem(self.x_nodes,
                                       self.y_nodes,
                                       self.z_nodes,
                                       self.k_func,
                                       self.f_func)
            self.non_bc = outputs['non_bc']
            if configs.L_SHAPED:
                self.f_reverse_translator = outputs['f_reverse_translator']
            u_true = outputs['u']
            stiff_p = outputs['a_mat_p']
            f_p = outputs['loading_vector_p']
        else:
            if configs.DIMENSIONS == 2:
                self.non_bc = utils.get_non_bc(len(self.x_nodes), len(self.y_nodes))
            elif configs.DIMENSIONS == 3:
                self.non_bc = utils.get_non_bc(len(self.x_nodes), len(self.y_nodes),
                                               len(self.z_nodes)).flatten()


        if init_from_func or init_from_input:
            self.stiff_p = stiff_p
            l_p = np.tril(self.stiff_p, -1)
            d_p = np.diag(np.diag(self.stiff_p))
            u_p = np.triu(self.stiff_p, 1)
            lu_p = l_p + u_p
            ld_p = l_p + d_p
            if self.solver_method in ['Jacobi', 'CG', 'MG-J']:
                self.m_p = d_p
                self.n_p = lu_p
            elif self.solver_method in ['GS', 'MG-GS']:
                self.m_p = ld_p
                self.n_p = u_p
            else:
                raise ValueError('Incorrect solver method')
            self.m_inv_n_p = np.linalg.solve(self.m_p, self.n_p)
            self.g_mat = (np.eye(stiff_p.shape[0]) - configs.OMEGA *
                        np.linalg.solve(self.m_p, self.stiff_p))

        self.f_p = f_p
        self.m_inv_f_p = np.linalg.solve(self.m_p, self.f_p)

        if init_from_input or update_f:
            u_true = utils.pad_dirichlet(np.linalg.solve(self.stiff_p, self.f_p))

        self.u_true = u_true

        return

    @staticmethod
    def compute_sorted_eigenvalues_and_eigenvectors(matrix):
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        sorted_sequence = np.argsort(eigenvalues)[::-1]
        eigenvalues_sorted = eigenvalues[sorted_sequence]
        eigenvectors_sorted = eigenvectors[:,sorted_sequence]
        return eigenvalues_sorted, eigenvectors_sorted

    def iterate_once(self, u_approx):
        if configs.DIMENSIONS == 1:
            u_approx_p = u_approx[1:-1]
        else:
            u_approx_p = u_approx[self.non_bc]
        if configs.SOLVER_METHOD in ['Jacobi', 'GS', 'MG-J', 'MG-GS']:
            u_approx_p_star = \
                self.m_inv_f_p - self.m_inv_n_p @ u_approx_p
            new_u_approx_p = \
                u_approx_p + configs.OMEGA * (u_approx_p_star - u_approx_p)
            u_approx = utils.pad_dirichlet(new_u_approx_p)
        elif configs.SOLVER_METHOD == 'CG':
            alpha = (np.matmul(self.r_step.transpose(), self.r_step) /
                     np.matmul(self.p_step.transpose(),
                               np.matmul(self.stiff_p, self.p_step)))
            u_approx[1:-1] = u_approx[1:-1] + alpha * self.p_step
            new_r_step = self.r_step - alpha * np.matmul(self.stiff_p, self.p_step)
            beta = (np.matmul(new_r_step.transpose(), new_r_step) /
                    np.matmul(self.r_step.transpose(), self.r_step))
            self.p_step = new_r_step + beta * self.p_step
            self.r_step = new_r_step
        return u_approx


class DeepONetSolver():
    def __init__(self, k_func, model_path, logger):
        self.y_nodes_don = None
        self.z_nodes_don = None
        if configs.DIMENSIONS == 1:
            self.x_nodes_don = np.linspace(0, 1, configs.NUM_OF_ELEMENTS_DON + 1)
        else:
            self.x_nodes_don = np.linspace(0, 1, configs.NUM_OF_ELEMENTS_DON_X + 1)
            self.y_nodes_don = np.linspace(0, 1, configs.NUM_OF_ELEMENTS_DON_Y + 1)
            if configs.DIMENSIONS == 3:
                self.z_nodes_don = np.linspace(0, 1, configs.NUM_OF_ELEMENTS_DON_Z + 1)
        self.logger = logger
        self.deeponet = DeepONet(self.x_nodes_don, logger,
                                 y_nodes=self.y_nodes_don, z_nodes=self.z_nodes_don)
        if model_path is not None:
            self.deeponet.branch_net = torch_load(os.path.join(model_path, 'branch_model'))
            self.deeponet.trunk_net = torch_load(os.path.join(model_path, 'trunk_model'))
            self.deeponet.bias = torch_load(os.path.join(model_path, 'bias'))
        self.k_func = k_func

    def iterate_once(self, u_approx):
        residual = utils.compute_residual(self.stiff_p, self.f_p, u_approx, non_bc=self.non_bc)
        f_func = residual       # Translate vector-form f into function form
        if configs.L_SHAPED:
            f_func = f_func / self.f_reverse_translator
        delta_u = self.deeponet.infer(self.k_func[None, :], f_func[None, :])[0][0, :]
        if configs.DIMENSIONS > 1:
            delta_u = delta_u.flatten()
        return u_approx + delta_u


class IterativeSolver(NumericalSolver, DeepONetSolver):
    def __init__(self, k_func, f_func, u_init, logger, model_path=None,
                 init_stiff_p=None, init_f_p=None):
        NumericalSolver.__init__(self, k_func, f_func, u_init, logger,
                                 init_stiff_p, init_f_p)
        DeepONetSolver.__init__(self, k_func, model_path, logger)
        self.residuals = np.zeros((configs.NUM_OF_ITERATIONS + 1,
                                   self.u_init.shape[0]))
        self.errors = np.zeros((configs.NUM_OF_ITERATIONS + 1,
                                self.u_init.shape[0]))
        self.residual_norms = np.zeros((configs.NUM_OF_ITERATIONS + 1,))
        self.error_norms = np.zeros((configs.NUM_OF_ITERATIONS + 1,))
        self.modes_of_interest = np.array(configs.MODES_OF_INTEREST)
        self.modes_errors = np.zeros((configs.NUM_OF_ITERATIONS + 1,
                                      len(self.modes_of_interest)))
        self.numerical_time = \
            np.zeros((int(configs.NUM_OF_ITERATIONS / configs.NUMERICAL_TO_DON_RATIO),))

    def solve(self):
        u_approx = self.u_init
        start_time = time.time()
        tracemalloc.start()
        tqdm_or_none = lambda x: x if configs.SOLVER_METHOD in ['MG-J', 'MG-GS'] else tqdm(x)
        if configs.ITERATION_METHOD == 'Numerical':
            self.update_metrics(u_approx, 0)
            for index in tqdm_or_none(range(configs.NUM_OF_ITERATIONS)):
                u_approx = NumericalSolver.iterate_once(self, u_approx)
                self.update_metrics(u_approx, index + 1)
            self.u_approx = u_approx
        if configs.ITERATION_METHOD == 'Numerical_DeepONet_Single':
            self.update_metrics(u_approx, 0)
            u_approx = DeepONetSolver.iterate_once(self, u_approx)
            self.update_metrics(u_approx, 1)
            for index in tqdm_or_none(range(1, configs.NUM_OF_ITERATIONS)):
                u_approx = NumericalSolver.iterate_once(self, u_approx)
                self.update_metrics(u_approx, index + 1)
            self.u_approx = u_approx
        if configs.ITERATION_METHOD == 'Numerical_DeepONet_Hybrid':
            self.update_metrics(u_approx, 0)
            clocking_index = 0
            clocking_start = time.time()
            for index in tqdm_or_none(range(configs.NUM_OF_ITERATIONS)):
                if (index + 1) % configs.NUMERICAL_TO_DON_RATIO == 0:
                # if index % configs.NUMERICAL_TO_DON_RATIO == 0:
                    u_approx = DeepONetSolver.iterate_once(self, u_approx)
                    self.numerical_time[clocking_index] = \
                        time.time() - clocking_start
                    clocking_start = time.time()
                    clocking_index += 1
                else:
                    u_approx = NumericalSolver.iterate_once(self, u_approx)
                self.update_metrics(u_approx, index + 1)
            self.u_approx = u_approx
        if configs.ITERATION_METHOD == 'DeepONet':
            self.update_metrics(u_approx, 0)
            for index in tqdm_or_none(range(configs.NUM_OF_ITERATIONS)):
                u_approx = DeepONetSolver.iterate_once(self, u_approx)
                self.update_metrics(u_approx, index + 1)
            self.u_approx = u_approx
        total_time = time.time() - start_time
        if configs.SOLVER_METHOD not in ['MG-J', 'MG-GS']:
            self.logger.info('Total solver time (s): ' + str(total_time))
            self.logger.info('Peak RAM traced (mb?): ' + str(tracemalloc.get_traced_memory()[1] / 1000000))
        tracemalloc.stop()

    def update_metrics(self, u_approx, index):
        residual = utils.compute_residual(self.stiff_p, self.f_p, u_approx, non_bc=self.non_bc)
        self.residuals[index, :] = residual
        error = u_approx - self.u_true
        self.errors[index, :] = error
        self.residual_norms[index] = np.sqrt(np.mean(residual ** 2))
        self.error_norms[index] = np.sqrt(np.mean(error ** 2))
        if configs.DIMENSIONS == 1:
            scores = np.linalg.solve(self.eigenvectors, self.errors[index, 1:-1])
        else:
            scores = np.linalg.solve(self.eigenvectors, self.errors[index, self.non_bc])
        self.modes_errors[index, :] = \
            scores[np.array(self.modes_of_interest)-1]

    def plot_metrics(self, save_path=None):
        utils.create_folder('outputs')
        length_history = self.residuals.shape[0]
        index_history = np.linspace(0, length_history-1, length_history)

        if configs.L_SHAPED:
            nodes, simplices, _, _, _, _ = utils.l_shaped_mesh()
        elif configs.DIMENSIONS == 3:
            xv, yv, zv = np.meshgrid(self.x_nodes, self.y_nodes, self.z_nodes)

        fig = plt.figure(figsize=(6.4 * 2, 4.8 * 3))

        if configs.DIMENSIONS == 1:
            plt.subplot(321)
            plt.plot(self.x_nodes, self.u_true, 'r')
            plt.plot(self.x_nodes, self.u_init, 'b')
            plt.plot(self.x_nodes, self.u_approx, 'c')
            plt.title('Comparison of Solutions')
            plt.legend(['u true', 'u init', 'u approx'])
            plt.xlabel('x')
            plt.xlabel('u')
        elif configs.DIMENSIONS == 2:
            plt.subplot(321)
            if configs.L_SHAPED:
                plt.tricontourf(nodes[:, 0], nodes[:, 1], simplices, self.u_true,
                                levels=np.linspace(np.min(self.u_true),
                                                   np.max(self.u_true), 100))
                plt.colorbar()
                plt.axis('equal')
                plt.title('True solution')
                plt.xlabel('x')
                plt.ylabel('y')
            else:
                u_true = np.reshape(self.u_true, (len(self.x_nodes),
                                                len(self.y_nodes)))
                contours = plt.contour(u_true,
                                    extent=[0, 1, 0, 1])
                plt.clabel(contours, inline=True, fontsize=8)
                plt.imshow(u_true, extent=[0, 1, 0, 1], origin='lower', alpha=0.5)
                plt.colorbar()
                plt.title('True solution')
                plt.xlabel('y')
                plt.ylabel('x')
        elif configs.DIMENSIONS == 3:
            ax = fig.add_subplot(3, 2, 1, projection='3d')
            u_true = np.reshape(self.u_true, (len(self.x_nodes),
                                              len(self.y_nodes),
                                              len(self.z_nodes)))
            utils.plot_3d(xv, yv, zv, u_true, ax)
            plt.title('True solution')

        
        if configs.DIMENSIONS == 1:
            plt.subplot(322)
            temp = np.copy(self.residuals[-1])
            plt.plot(self.x_nodes, temp, '-b')
            plt.plot(self.x_nodes, temp*0.0, '--k')
            plt.title('Residual Upon Completion')
            plt.legend(['residual','zero'])
            plt.xlabel('x')
            plt.ylabel('Residual Vector')
        elif configs.DIMENSIONS == 2:
            plt.subplot(322)
            if configs.L_SHAPED:
                plt.tricontourf(nodes[:, 0], nodes[:, 1], simplices, self.u_approx,
                                levels=np.linspace(np.min(self.u_true),
                                                   np.max(self.u_true), 100))
                plt.colorbar()
                plt.axis('equal')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('Approximate solution')
            else:
                u_approx = np.reshape(self.u_approx, (len(self.x_nodes),
                                                    len(self.y_nodes)))
                contours = plt.contour(u_approx,
                                    extent=[0, 1, 0, 1])
                plt.clabel(contours, inline=True, fontsize=8)
                plt.imshow(u_approx, extent=[0, 1, 0, 1], origin='lower', alpha=0.5)
                plt.colorbar()
                plt.xlabel('y')
                plt.ylabel('x')
                plt.title('Approximate solution')
        elif configs.DIMENSIONS == 3:
            ax = fig.add_subplot(3, 2, 2, projection='3d')
            u_approx = np.reshape(self.u_approx, (len(self.x_nodes),
                                                  len(self.y_nodes),
                                                  len(self.z_nodes)))
            utils.plot_3d(xv, yv, zv, u_approx, ax)
            plt.title('Approximate solution')

        plt.subplot(323)
        if configs.L_SHAPED:
            plt.semilogy(index_history, self.error_norms)
            plt.title('History of Error Norm')
            plt.xlabel('Number of iterations')
            plt.ylabel('Error Norm')
        else:
            plt.semilogy(index_history, self.residual_norms)
            plt.title('History of Residual Norm')
            plt.xlabel('Number of iterations')
            plt.ylabel('Residual Norm')

        if configs.DIMENSIONS == 1:
            plt.subplot(324)
            plt.semilogy(index_history, self.error_norms)
            plt.title('History of Error Norm')
            plt.xlabel('Number of iterations')
            plt.ylabel('Error Norm')
        elif configs.DIMENSIONS == 2:
            plt.subplot(324)
            if configs.L_SHAPED:
                u_error = np.abs(self.u_approx - self.u_true)
                plt.tricontourf(nodes[:, 0], nodes[:, 1], simplices,
                                u_error,
                                levels=np.linspace(np.min(u_error),
                                                   np.max(u_error), 100))
                plt.colorbar()
                plt.axis('equal')
                plt.title('Error')
                plt.xlabel('x')
                plt.ylabel('y')
            else:
                contours = plt.contour(np.abs(u_approx - u_true),
                                       extent=[0, 1, 0, 1])
                plt.clabel(contours, inline=True, fontsize=8)
                plt.imshow(np.abs(u_approx - u_true), extent=[0, 1, 0, 1],
                           origin='lower', alpha=0.5)
                plt.colorbar()
                plt.title('Error')
                plt.xlabel('y')
                plt.ylabel('x')
        elif configs.DIMENSIONS == 3:
            # ax = fig.add_subplot(3, 2, 4, projection='3d')
            # u_error = np.reshape(np.abs(u_approx - u_true), (len(self.x_nodes),
            #                                                  len(self.y_nodes),
            #                                                  len(self.z_nodes)))
            # utils.plot_3d(xv, yv, zv, u_error, ax)
            # plt.title('Abs error')
            plt.subplot(324)
            plt.semilogy(index_history, self.error_norms)
            plt.title('History of Error Norm')
            plt.xlabel('Number of iterations')
            plt.ylabel('Error Norm')

        plt.subplot(325)
        for index_mode, mode in enumerate(self.modes_of_interest):
            plt.semilogy(index_history, np.abs(self.modes_errors[:, index_mode]),
                         label='Mode '+ str(mode))
            plt.legend()
        plt.title('Mode-wise Abs Error')

        if configs.DIMENSIONS == 1:
            plt.subplot(326)
            plt.plot(self.numerical_time)
            plt.title('Numerical solver iteration batch elapsed time')
            plt.xlabel('Batch number')
            plt.ylabel('Time (s)')
        elif configs.DIMENSIONS == 2:
            plt.subplot(326)
            contours = plt.contour(self.k_func, extent=[0, 1, 0, 1])
            plt.clabel(contours, inline=True, fontsize=8)
            plt.imshow(self.k_func, extent=[0, 1, 0, 1], origin='lower', alpha=0.5)
            plt.colorbar()
            plt.title('k(x,y)')
            plt.xlabel('y')
            plt.ylabel('x')
        elif configs.DIMENSIONS == 3:
            ax = fig.add_subplot(3, 2, 6, projection='3d')
            utils.plot_3d(xv, yv, zv, self.k_func, ax)
            plt.title('k(x,y,z)')

        if save_path is None:
            save_path = os.path.join('outputs', 'iterative_solver_outputs')
        plt.tight_layout()
        plt.savefig(save_path)


class MultiGrid():
    def __init__(self, k_func, f_func, u_init, logger, model_path=None, levels=None):
        self.levels = levels
        if levels is None:
            # 2^? elements in the deepest level
            self.levels = int(np.log2(configs.NUM_OF_ELEMENTS)) - 3
        self.solvers = [IterativeSolver(k_func, f_func, u_init, logger, model_path)]
        self.restrictors = []
        self.prolongators = []

        # object attributes from numerical solver
        self.stiff_p = self.solvers[0].stiff_p
        self.f_p = self.solvers[0].f_p
        self.u_init = u_init
        self.u_true = self.solvers[0].u_true
        self.eigenvalues = self.solvers[0].eigenvalues
        self.eigenvectors = self.solvers[0].eigenvectors
        self.logger = logger
        self.u_approx = None

        if configs.DIMENSIONS == 1:
            self.x_nodes = np.linspace(0, 1, u_init.shape[0])
            self.y_nodes = None
        elif configs.DIMENSIONS == 2:
            self.x_nodes = np.linspace(0, 1, k_func.shape[0])
            self.y_nodes = np.linspace(0, 1, k_func.shape[1])
        elif configs.DIMENSIONS == 3:
            self.x_nodes = np.linspace(0, 1, k_func.shape[0])
            self.y_nodes = np.linspace(0, 1, k_func.shape[1])
            self.z_nodes = np.linspace(0, 1, k_func.shape[2])

        # object attributes from iterative solver
        self.residuals = np.zeros((configs.NUM_OF_MG_V_CYCLE + 1,
                                   u_init.shape[0]))
        self.errors = np.zeros((configs.NUM_OF_MG_V_CYCLE + 1,
                                u_init.shape[0]))
        self.residual_norms = np.zeros((configs.NUM_OF_MG_V_CYCLE + 1,))
        self.error_norms = np.zeros((configs.NUM_OF_MG_V_CYCLE + 1,))
        self.modes_of_interest = \
            np.array(configs.MODES_OF_INTEREST)[np.array(configs.MODES_OF_INTEREST) < k_func.shape[0]-1]
        self.modes_errors = np.zeros((configs.NUM_OF_MG_V_CYCLE + 1,
                                      len(self.modes_of_interest)))
        self.numerical_time = \
            np.zeros((int(configs.NUM_OF_MG_V_CYCLE / configs.NUMERICAL_TO_DON_RATIO),))

        self.cycle_history = []

        stiff_p, loading_vector_p = self.stiff_p, np.zeros_like(self.f_p)
        for _ in range(1, self.levels + 1):
            restrictor, prolongator = \
                    self.create_restrictor_prolongator(k_func.shape)
            self.restrictors.append(restrictor)
            self.prolongators.append(prolongator)

            loading_vector_p = restrictor @ loading_vector_p
            stiff_p = restrictor @ stiff_p @ prolongator
            # TODO: get rid of the interpolation of k_func
            if configs.DIMENSIONS == 1:
                k_func = utils.pad_dirichlet(restrictor @ k_func[1:-1],
                                             start=k_func[0],
                                             end=k_func[-1])
            elif configs.DIMENSIONS == 2:
                k_func = k_func[::2, ::2]
            elif configs.DIMENSIONS == 3:
                k_func = k_func[::2, ::2, ::2]
            next_solver = IterativeSolver(k_func, None,
                                          utils.pad_dirichlet(loading_vector_p),
                                          logger,
                                          model_path,
                                          init_stiff_p=stiff_p,
                                          init_f_p=loading_vector_p)
            self.solvers.append(next_solver)

    @staticmethod
    def create_restrictor_prolongator(size):
        # size: including [0] and [-1], for the bigger matrix/vector
        if configs.DIMENSIONS == 1:
            size_reduced = int((size[0] - 2) / 2)
            restrictor = np.zeros((size_reduced, size))
            prolongator = np.zeros((size, size_reduced))
            for i in range(size_reduced):
                prolongator[2 * i:2 * i + 3, i] = [0.5, 1.0, 0.5]
                restrictor[i, 2 * i:2 * i + 3] = [0.25, 0.5, 0.25]
        elif configs.DIMENSIONS == 2:
            n_x_inner, n_y_inner = size[0] - 2, size[1] - 2
            n_x_reduced, n_y_reduced = int(n_x_inner / 2), int(n_y_inner / 2)
            restrictor = np.zeros((n_x_reduced, n_y_reduced, n_x_inner, n_y_inner))
            prolongator = np.zeros((n_x_reduced, n_y_reduced, n_x_inner, n_y_inner))
            restrictor_mid, restrictor_cross, restrictor_scat = 1 / 4, 1 / 8, 1 / 16
            prolongator_mid, prolongator_cross, prolongator_scat = 1, 1 / 2, 1 / 4
            for row_index in range(n_x_reduced):
                for col_index in range(n_y_reduced):
                    restrictor[row_index, col_index, 2 * row_index + 1, 2 * col_index + 1] = restrictor_mid
                    restrictor[row_index, col_index, 2 * row_index, 2 * col_index + 1] = restrictor_cross
                    restrictor[row_index, col_index, 2 * row_index + 1, 2 * col_index] = restrictor_cross
                    restrictor[row_index, col_index, 2 * row_index + 2, 2 * col_index + 1] = restrictor_cross
                    restrictor[row_index, col_index, 2 * row_index + 1, 2 * col_index + 2] = restrictor_cross
                    restrictor[row_index, col_index, 2 * row_index, 2 * col_index] = restrictor_scat
                    restrictor[row_index, col_index, 2 * row_index, 2 * col_index + 2] = restrictor_scat
                    restrictor[row_index, col_index, 2 * row_index + 2, 2 * col_index] = restrictor_scat
                    restrictor[row_index, col_index, 2 * row_index + 2, 2 * col_index + 2] = restrictor_scat

                    prolongator[row_index, col_index, 2 * row_index + 1, 2 * col_index + 1] = prolongator_mid
                    prolongator[row_index, col_index, 2 * row_index, 2 * col_index + 1] = prolongator_cross
                    prolongator[row_index, col_index, 2 * row_index + 1, 2 * col_index] = prolongator_cross
                    prolongator[row_index, col_index, 2 * row_index + 2, 2 * col_index + 1] = prolongator_cross
                    prolongator[row_index, col_index, 2 * row_index + 1, 2 * col_index + 2] = prolongator_cross
                    prolongator[row_index, col_index, 2 * row_index, 2 * col_index] = prolongator_scat
                    prolongator[row_index, col_index, 2 * row_index, 2 * col_index + 2] = prolongator_scat
                    prolongator[row_index, col_index, 2 * row_index + 2, 2 * col_index] = prolongator_scat
                    prolongator[row_index, col_index, 2 * row_index + 2, 2 * col_index + 2] = prolongator_scat
            restrictor = np.reshape(restrictor, (n_x_reduced * n_y_reduced, n_x_inner * n_y_inner))
            prolongator = np.reshape(prolongator.transpose(), (n_x_inner * n_y_inner, n_x_reduced * n_y_reduced))
        elif configs.DIMENSIONS == 3:
            n_x_inner, n_y_inner, n_z_inner = size[0] - 2, size[1] - 2, size[2] - 2
            n_x_reduced = int(n_x_inner / 2)
            n_y_reduced = int(n_y_inner / 2)
            n_z_reduced = int(n_z_inner / 2)
            restrictor = np.zeros((n_x_reduced, n_y_reduced, n_z_reduced,
                                   n_x_inner, n_y_inner, n_z_inner))
            prolongator = np.zeros((n_x_reduced, n_y_reduced, n_z_reduced,
                                    n_x_inner, n_y_inner, n_z_inner))
            restrictor_mid, restrictor_cross, restrictor_scat, restrictor_ext = \
                1 / 8, 1 / 16, 1 / 32, 1 / 64
            prolongator_mid, prolongator_cross, prolongator_scat, prolongator_ext = \
                1, 1 / 2, 1 / 4, 1 / 8
            cross_index_increments = np.array([[0, 1, 1],
                                               [1, 0, 1],
                                               [1, 1, 0],
                                               [2, 1, 1],
                                               [1, 2, 1],
                                               [1, 1, 2]])
            scatter_index_increments = np.array([[0, 0, 1],
                                                 [0, 1, 0],
                                                 [1, 0, 0],
                                                 [2, 2, 1],
                                                 [2, 1, 2],
                                                 [1, 2, 2],
                                                 [0, 2, 1],
                                                 [0, 1, 2],
                                                 [1, 0, 2],
                                                 [2, 0, 1],
                                                 [2, 1, 0],
                                                 [1, 2, 0]])
            ext_index_increments = np.array([[0, 0, 0],
                                             [1, 1, 1],
                                             [1, 1, 1],
                                             [1, 1, 1],
                                             [1, 1, 1],
                                             [1, 1, 1],
                                             [1, 1, 1],
                                             [1, 1, 1]])
            for row_index in range(n_x_reduced):
                for col_index in range(n_y_reduced):
                    for depth_index in range(n_z_reduced):
                        restrictor[row_index, col_index, depth_index,
                                   2 * row_index + 1, 2 * col_index + 1,
                                   2 * depth_index + 1] = restrictor_mid
                        prolongator[row_index, col_index, depth_index,
                                    2 * row_index + 1, 2 * col_index + 1,
                                    2 * depth_index + 1] = prolongator_mid
                        for increment in cross_index_increments:
                            restrictor[row_index, col_index, depth_index,
                                       2 * row_index + increment[0],
                                       2 * col_index + increment[1],
                                       2 * depth_index + increment[2]] = restrictor_cross
                            prolongator[row_index, col_index, depth_index,
                                        2 * row_index + increment[0],
                                        2 * col_index + increment[1],
                                        2 * depth_index + increment[2]] = prolongator_cross
                        for increment in scatter_index_increments:
                            restrictor[row_index, col_index, depth_index,
                                       2 * row_index + increment[0],
                                       2 * col_index + increment[1],
                                       2 * depth_index + increment[2]] = restrictor_scat
                            prolongator[row_index, col_index, depth_index,
                                        2 * row_index + increment[0],
                                        2 * col_index + increment[1],
                                        2 * depth_index + increment[2]] = prolongator_scat
                        for increment in ext_index_increments:
                            restrictor[row_index, col_index, depth_index,
                                       2 * row_index + increment[0],
                                       2 * col_index + increment[1],
                                       2 * depth_index + increment[2]] = restrictor_ext
                            prolongator[row_index, col_index, depth_index,
                                        2 * row_index + increment[0],
                                        2 * col_index + increment[1],
                                        2 * depth_index + increment[2]] = prolongator_ext
            restrictor = np.reshape(restrictor, (n_x_reduced * n_y_reduced * n_z_reduced,
                                                 n_x_inner * n_y_inner * n_z_inner))
            prolongator = np.reshape(prolongator.transpose(), (n_x_inner * n_y_inner * n_z_inner,
                                                               n_x_reduced * n_y_reduced * n_z_reduced))
        return restrictor, prolongator

    def smoothing(self, u_approx, res_p, level=0):
        self.cycle_history.append(-level)
        self.solvers[level].u_init = u_approx
        if level > 0:
            self.solvers[level].assemble(f_p=res_p)
        self.solvers[level].solve()
        return self.solvers[level].u_approx

    def v_cycle(self, e, res_p, level=0):
        # When level=0, e is from previous iteration results, res_p is f_p
        # When level>0, e is initialized to be 0, res_p is calculated residual
        e = self.smoothing(e, res_p, level)
        if level > self.levels - 1:     # only when self.levels=0; single grid
            e = self.smoothing(e, res_p, level)
            return e

        resres_p = utils.compute_residual(self.solvers[level].stiff_p,
                                          res_p, e,
                                          pad=False,
                                          non_bc=self.solvers[level].non_bc)

        restricted_resres_p = self.restrictors[level] @ resres_p
        ee = utils.pad_dirichlet(np.zeros_like(restricted_resres_p))
        
        if level == self.levels - 1:
            ee = self.smoothing(ee, restricted_resres_p, level=level+1)
            # ee = utils.pad_Dirichlet(np.linalg.solve(
            #         self.solvers[level+1].stiff_p, restricted_resres_p))
        else:
            ee = self.v_cycle(ee, restricted_resres_p, level=level+1)
        if configs.DIMENSIONS == 1:
            e = e + utils.pad_dirichlet(self.prolongators[level] @ ee[1:-1])
        else:
            e = e + utils.pad_dirichlet(self.prolongators[level] @ ee[self.solvers[level + 1].non_bc])
        e = self.smoothing(e, res_p, level)
        return e

    def update_metrics(self, u_approx, index):
        residual = utils.compute_residual(self.stiff_p, self.f_p, u_approx,
                                          non_bc=self.solvers[0].non_bc)
        self.residuals[index, :] = residual
        error = u_approx - self.u_true
        self.errors[index, :] = error
        self.residual_norms[index] = np.sqrt(np.mean(residual ** 2))
        self.error_norms[index] = np.sqrt(np.mean(error ** 2))
        if configs.DIMENSIONS == 1:
            scores = np.linalg.solve(self.eigenvectors, self.errors[index, 1:-1])
        else:
            scores = np.linalg.solve(self.eigenvectors,
                                     self.errors[index, self.solvers[0].non_bc])
        self.modes_errors[index, :] = \
            scores[np.array(self.modes_of_interest)-1]

    def solve(self):
        u_approx = self.u_init
        start_time = time.time()
        tracemalloc.start()

        self.update_metrics(u_approx, 0)
        for id_v in tqdm(range(configs.NUM_OF_MG_V_CYCLE)):
            u_approx = self.v_cycle(u_approx, self.f_p)
            self.update_metrics(u_approx, id_v + 1)
        self.u_approx = u_approx

        total_time = time.time() - start_time
        self.logger.info('Total solver time (s): ' + str(total_time))
        self.logger.info('Peak RAM traced (mb?): ' + str(tracemalloc.get_traced_memory()[1] / 1000000))
        tracemalloc.stop()

    def plot_metrics(self, save_path=None):
        utils.create_folder('outputs')
        length_history = self.residuals.shape[0]
        index_history = np.linspace(0, length_history-1, length_history)

        fig = plt.figure(figsize=(6.4 * 2, 4.8 * 3))
        if configs.DIMENSIONS == 3:
            xv, yv, zv = np.meshgrid(self.x_nodes, self.y_nodes, self.z_nodes)

        # TODO: figure out code duplication with father
        if configs.DIMENSIONS == 1:
            plt.subplot(321)
            plt.plot(self.x_nodes, self.u_true, 'r')
            plt.plot(self.x_nodes, self.u_init, 'b')
            plt.plot(self.x_nodes, self.u_approx, 'c')
            plt.title('Comparison of Solutions')
            plt.legend(['u true', 'u init', 'u approx'])
            plt.xlabel('x')
            plt.xlabel('u')
        elif configs.DIMENSIONS == 2:
            plt.subplot(321)
            u_true = np.reshape(self.u_true, (len(self.x_nodes),
                                              len(self.y_nodes)))
            contours = plt.contour(u_true,
                                   extent=[0, 1, 0, 1])
            plt.clabel(contours, inline=True, fontsize=8)
            plt.imshow(u_true, extent=[0, 1, 0, 1], origin='lower', alpha=0.5)
            plt.colorbar()
            plt.title('True solution')
            plt.xlabel('y')
            plt.ylabel('x')
        elif configs.DIMENSIONS == 3:
            ax = fig.add_subplot(3, 2, 1, projection='3d')
            u_true = np.reshape(self.u_true, (len(self.x_nodes),
                                              len(self.y_nodes),
                                              len(self.z_nodes)))
            utils.plot_3d(xv, yv, zv, u_true, ax)
            plt.title('True solution')

        if configs.DIMENSIONS == 1:
            plt.subplot(322)
            temp = np.copy(self.residuals[-1])
            plt.plot(self.x_nodes, temp, '-b')
            plt.plot(self.x_nodes, temp*0.0, '--k')
            plt.title('Residual Upon Completion')
            plt.legend(['residual','zero'])
            plt.xlabel('x')
            plt.ylabel('Residual Vector')
        elif configs.DIMENSIONS == 2:
            plt.subplot(322)
            u_approx = np.reshape(self.u_approx, (len(self.x_nodes),
                                                  len(self.y_nodes)))
            contours = plt.contour(u_approx,
                                   extent=[0, 1, 0, 1])
            plt.clabel(contours, inline=True, fontsize=8)
            plt.imshow(u_approx, extent=[0, 1, 0, 1], origin='lower', alpha=0.5)
            plt.colorbar()
            plt.xlabel('y')
            plt.ylabel('x')
            plt.title('Approximate solution')
        elif configs.DIMENSIONS == 3:
            ax = fig.add_subplot(3, 2, 2, projection='3d')
            u_approx = np.reshape(self.u_approx, (len(self.x_nodes),
                                                  len(self.y_nodes),
                                                  len(self.z_nodes)))
            utils.plot_3d(xv, yv, zv, self.u_approx, ax)
            plt.title('Approximate solution')

        plt.subplot(323)
        plt.semilogy(index_history, self.residual_norms)
        plt.title('History of Residual Norm')
        plt.xlabel('Number of V cycles')
        plt.ylabel('Residual Norm')

        plt.subplot(324)
        if configs.DIMENSIONS == 1 or configs.DIMENSIONS == 3:
            plt.semilogy(index_history, self.error_norms)
            plt.title('History of Error Norm')
            plt.xlabel('Number of iterations')
            plt.ylabel('Error Norm')
        elif configs.DIMENSIONS == 2:
            contours = plt.contour(np.abs(u_approx - u_true),
                                   extent=[0, 1, 0, 1])
            plt.clabel(contours, inline=True, fontsize=8)
            plt.imshow(np.abs(u_approx - u_true), extent=[0, 1, 0, 1],
                       origin='lower', alpha=0.5)
            plt.colorbar()
            plt.title('Error')
            plt.xlabel('y')
            plt.ylabel('x')

        plt.subplot(325)
        for index_mode, mode in enumerate(self.modes_of_interest):
            plt.semilogy(index_history, np.abs(self.modes_errors[:, index_mode]),
                         label='Mode '+ str(mode))
            plt.legend()
        plt.title('Mode-wise Abs Error')

        plt.subplot(326)
        plt.plot(self.cycle_history,'-^b')
        plt.xlabel('Smoothing Step')
        plt.ylabel('Negative Level')

        # TODO: time consumption
        # plt.subplot(326)
        # plt.plot(self.numerical_time)
        # plt.title('Numerical solver iteration batch elapsed time')
        # plt.xlabel('Batch number')
        # plt.ylabel('Time (s)')

        if save_path is None:
            save_path = os.path.join('outputs', 'MG_solver_outputs')
        plt.tight_layout()
        plt.savefig(save_path)
