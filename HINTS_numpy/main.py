import os
import numpy as np
from data_handler import DataHandler
from iterative_solver import IterativeSolver, MultiGrid
from deeponet import DeepONet
from utils import logger, init_simple
import configs
from scipy.interpolate import interp2d, interpn


logger.setLevel(15)


def select_test_sample(test_sample_num=None):
    k, f, u_init = init_simple()
    if test_sample_num is None:
        return k, f, u_init
    if configs.PROBLEM == 'helmholtz' or configs.L_SHAPED:
        data_handler = DataHandler(logger)
        data = data_handler.get_data(percentage_test=configs.PERCENTAGE_TEST,
                                     force_create=False)
        k = data['k_test'][test_sample_num, :]

        if configs.DIMENSIONS == 1:
            if f.shape[0] != k.shape[0]:
                k = np.interp(np.linspace(0.0, 1.0, configs.NUM_OF_ELEMENTS + 1),
                              np.linspace(0, 1, k.shape[0]), k)
        if configs.DIMENSIONS == 2:
            f = data['f_test'][test_sample_num, :]
            if f.shape[0] != (k.shape[0] * k.shape[1]):
                k_interp = interp2d(data['x_nodes'], data['y_nodes'], k)
                k = k_interp(np.linspace(0.0, 1.0, configs.NUM_OF_ELEMENTS_X + 1),
                             np.linspace(0.0, 1.0, configs.NUM_OF_ELEMENTS_Y + 1))
                f_interp = interp2d(data['x_nodes'], data['y_nodes'], f)
                f = f_interp(np.linspace(0.0, 1.0, configs.NUM_OF_ELEMENTS_X + 1),
                             np.linspace(0.0, 1.0, configs.NUM_OF_ELEMENTS_Y + 1)).flatten()
        if configs.DIMENSIONS == 3:
            if f.shape[0] != (k.shape[0] * k.shape[1] * k.shape[2]):
                f = data['f_test'][test_sample_num, :]
                deeponet_points = (data['x_nodes'], data['y_nodes'], data['z_nodes'])
                x_nodes = np.linspace(0.0, 1.0, configs.NUM_OF_ELEMENTS_X + 1)
                y_nodes = np.linspace(0.0, 1.0, configs.NUM_OF_ELEMENTS_Y + 1)
                z_nodes = np.linspace(0.0, 1.0, configs.NUM_OF_ELEMENTS_Z + 1)
                xv, yv, zv = np.meshgrid(x_nodes, y_nodes, z_nodes)
                grid_points = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1)
                k = np.reshape(interpn(deeponet_points, k, grid_points),
                               (configs.NUM_OF_ELEMENTS_X + 1,
                                configs.NUM_OF_ELEMENTS_Y + 1,
                                configs.NUM_OF_ELEMENTS_Z + 1))
                f = interpn(deeponet_points, f, grid_points)
    return k, f, u_init


def run_numerical():
    k, f, u_init = select_test_sample(0)
    if configs.SOLVER_METHOD in ['MG-J', 'MG-GS']:
        solver = MultiGrid(k, f, u_init, logger, levels=configs.NUM_OF_LEVELS)
    else:
        solver = IterativeSolver(k, f, u_init, logger)
    solver.solve()
    solver.plot_metrics()
    return


def run_dl():
    data_handler = DataHandler(logger)
    data = data_handler.get_data(percentage_test=configs.PERCENTAGE_TEST,
                                 force_create=False)

    output_file_path = os.path.join('outputs', 'results_deeponet.npz')
    if configs.FORCE_RETRAIN or not os.path.exists(output_file_path):
        deeponet = DeepONet(data['x_nodes'], logger,
                            y_nodes=data['y_nodes'], z_nodes=data['z_nodes'])
        deeponet.fit(data, num_of_epochs=configs.EPOCHS,
                     batch_size=configs.BATCH_SIZE,
                     plot_interval=configs.PLOT_INTERVAL,
                     should_save=True)

        u_init = deeponet.infer(data['k_test'], data['f_test'],
                                output_file_path)[0][0, :]
    else:
        u_init = np.load(output_file_path)['u_pred'][0, :]

    model_path = os.path.join('models', configs.MODEL_NAME)
    iterative_solver = IterativeSolver(data['k_test'][0, :],
                                       data['f_test'][0, :],
                                       u_init, logger, model_path)
    iterative_solver.solve()
    iterative_solver.plot_metrics()
    return


def run_hybrid():
    k, f, u_init = select_test_sample(10)  # L-shaped 14, 2D 27, 3D 7, 2D new
    model_path = os.path.join('models', configs.MODEL_NAME)

    if configs.SOLVER_METHOD in ['MG-J', 'MG-GS']:
        logger.info('Running MG solver with Hybrid smoother')
        solver = MultiGrid(k, f, u_init, logger, model_path,
                           levels=configs.NUM_OF_LEVELS)
    else:
        logger.info('Running Hybrid solver')
        solver = IterativeSolver(k, f, u_init, logger, model_path)
    solver.solve()
    solver.plot_metrics()
    return


if __name__ == '__main__':
    if configs.ITERATION_METHOD == 'Numerical':
        run_numerical()
    elif (configs.ITERATION_METHOD == 'DeepONet' or
          configs.ITERATION_METHOD == 'Numerical_DeepONet_Single'):
        run_dl()
    elif configs.ITERATION_METHOD == 'Numerical_DeepONet_Hybrid':
        run_hybrid()
    else:
        raise ValueError('Incorrect iteration method')
    print('End main')
