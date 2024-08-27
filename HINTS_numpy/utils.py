import logging
import numpy as np
import configs
import torch
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Delaunay
from scipy.io import loadmat
from scipy.interpolate import interp2d


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s, %(levelname)s:     %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S')

logger = logging.getLogger()


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


def to_numpy(x):
    if isinstance(x, list):
        return [to_numpy(i) for i in x]
    else:
        return x.cpu().detach().numpy().astype('float64')


def to_torch(x):
    return torch.tensor(x, dtype=torch.float, device=configs.DEVICE)


# TODO: convert to torch batch loader
def batch_loader(num_of_train_samples, batch_size):
    num_of_batches = np.int64(np.ceil(num_of_train_samples / batch_size))
    data = np.zeros((num_of_batches, 3), dtype=np.int64)      # start, end, length
    data[:, 0] = np.arange(0, num_of_train_samples, batch_size)
    data[:, 1] = data[:, 0] + batch_size
    data[:, 2] = batch_size
    data[-1, 1] = num_of_train_samples
    data[-1, 2] = num_of_train_samples - data[-1,0]
    return data, data.shape[0]


def results_plotter(epoch_index, num_of_samples_test,
                    x_nodes_test, u_test_pred, u_test,
                    loss_train, loss_test):
    create_folder(os.path.join('debug_figs', 'test'))
    create_folder(os.path.join('debug_figs', 'loss'))
    num_of_test_plots = 16
    num_of_samples_per_row = 2
    num_rows = (num_of_test_plots - 1) // num_of_samples_per_row + 1
    num_cols = 2 * (num_of_samples_per_row + int(configs.DIMENSIONS > 1))
    if configs.DIMENSIONS == 3:
        fig = plt.figure(figsize=(3.2 * num_cols, 2.4 * num_rows))  # TODO: Fix for other dims
    else:
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols,
                                figsize=(3.2 * num_cols, 2.4 * num_rows))
    for test_sample_index in range(num_of_samples_test):  # TODO: Unecessary loop
        # a, b, theta_deg = geo_test[itest]
        # theta = theta_deg*np.pi/180
        # x, y = np.meshgrid(coords,coords)
        # cos = np.cos(theta)
        # sin = np.sin(theta)
        # xp = x*cos+y*sin
        # yp = -x*sin+y*cos
        # not_solid = (xp**2/a**2+yp**2/b**2-1<0)
    
        if test_sample_index < num_of_test_plots:
            if configs.DIMENSIONS == 1:
                ax = axs[test_sample_index // num_of_samples_per_row,
                         2 * (test_sample_index % num_of_samples_per_row)]
                ax.plot(x_nodes_test, u_test_pred[test_sample_index],'-b')
                ax.set_title('Test # ' + str(test_sample_index + 1) + ' Pred')

                #ax = axs[itest//num_sample_per_row,3*(itest%num_sample_per_row)+1]
                ax.plot(x_nodes_test, u_test[test_sample_index], '-r')
                #ax.set_title('Test # {} True'.format(itest+1))

                ax = axs[test_sample_index // num_of_samples_per_row,
                         2 * (test_sample_index % num_of_samples_per_row) + 1]
                ax.plot(x_nodes_test, u_test_pred[test_sample_index] -
                                        u_test[test_sample_index])
            elif configs.DIMENSIONS == 2:
                if configs.L_SHAPED:
                    points, simplices, _, _, _, _ = l_shaped_mesh()
                else:
                    x_all = np.linspace(0.0, 1.0, configs.NUM_OF_ELEMENTS_DON_X + 1)
                    y_all = np.linspace(0.0, 1.0, configs.NUM_OF_ELEMENTS_DON_Y + 1)
                ax = axs[test_sample_index // num_of_samples_per_row,
                         3 * (test_sample_index % num_of_samples_per_row)]
                z = u_test_pred[test_sample_index]

                if configs.L_SHAPED:
                    cs = ax.tricontourf(points[:, 0], points[:, 1], simplices, z,
                                        levels=np.linspace(np.min(z), np.max(z), 100))
                else:
                    cs = ax.contourf(y_all, x_all, z)
                divider = make_axes_locatable(ax)
                ax_cb = divider.new_horizontal(size="5%", pad=0.05)
                fig = ax.get_figure()
                fig.add_axes(ax_cb)
                ticks = np.linspace(np.min(z),np.max(z),8)
                cbar = fig.colorbar(cs, cax=ax_cb, ticks=ticks)
                ax_cb.yaxis.tick_right()
                ax.set_title('Test # ' + str(test_sample_index + 1) + ' Pred')
                ax.set_xlabel('y')
                ax.set_ylabel('x')
                ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
                ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
                ax.set_aspect('equal', 'box')

                ax = axs[test_sample_index // num_of_samples_per_row,
                         3 * (test_sample_index % num_of_samples_per_row) + 1]
                z = u_test[test_sample_index]
                if configs.L_SHAPED:
                    cs = ax.tricontourf(points[:, 0], points[:, 1], simplices, z,
                                    levels=np.linspace(np.min(z), np.max(z), 100))
                else:
                    cs = ax.contourf(y_all, x_all, z)
                divider = make_axes_locatable(ax)
                ax_cb = divider.new_horizontal(size="5%", pad=0.05)
                fig = ax.get_figure()
                fig.add_axes(ax_cb)
                ticks = np.linspace(np.min(z),np.max(z),8)
                cbar = fig.colorbar(cs, cax=ax_cb, ticks=ticks)
                ax_cb.yaxis.tick_right()
                ax.set_title('Test # ' + str(test_sample_index + 1) + ' True')
                ax.set_xlabel('y')
                ax.set_ylabel('x')
                ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
                ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
                ax.set_aspect('equal', 'box')

                ax = axs[test_sample_index // num_of_samples_per_row,
                         3 * (test_sample_index % num_of_samples_per_row)+ 2]
                z = u_test_pred[test_sample_index] - u_test[test_sample_index]
                if configs.L_SHAPED:
                    cs = ax.tricontourf(points[:, 0], points[:, 1], simplices, z,
                                    levels=np.linspace(np.min(z), np.max(z), 100))
                else:
                    cs = ax.contourf(y_all, x_all, z)
                divider = make_axes_locatable(ax)
                ax_cb = divider.new_horizontal(size="5%", pad=0.05)
                fig = ax.get_figure()
                fig.add_axes(ax_cb)
                ticks = np.linspace(np.min(z),np.max(z),8)
                cbar = fig.colorbar(cs, cax=ax_cb, ticks=ticks)
                ax_cb.yaxis.tick_right()
                l2_error = np.sqrt(np.nanmean((u_test_pred[test_sample_index] -
                                               u_test[test_sample_index]) ** 2))
                l2_relative_error = l2_error / np.sqrt(np.nanmean(u_test[test_sample_index] ** 2))
                ax.set_title('L2 Rel. Err. ' + str(l2_relative_error))
                ax.set_xlabel('y')
                ax.set_ylabel('x')
                ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
                ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
                ax.set_aspect('equal', 'box')
            elif configs.DIMENSIONS == 3:
                # Take this out
                x_all = np.linspace(0.0, 1.0, configs.NUM_OF_ELEMENTS_DON_X + 1)
                y_all = np.linspace(0.0, 1.0, configs.NUM_OF_ELEMENTS_DON_Y + 1)
                z_all = np.linspace(0.0, 1.0, configs.NUM_OF_ELEMENTS_DON_Z + 1)
                xv, yv, zv = np.meshgrid(x_all, y_all, z_all)

                ax = fig.add_subplot(num_rows, num_cols,test_sample_index * 3 + 1,
                                     projection='3d')
                z = u_test_pred[test_sample_index]

                plot_3d(xv, yv, zv, z, ax)
                divider = make_axes_locatable(ax)
                ax_cb = divider.new_horizontal(size="5%", pad=0.05)
                # ticks = np.linspace(np.min(z), np.max(z), 8)
                # ax.colorbar(cax=ax_cb, ticks=ticks)
                # ax_cb.yaxis.tick_right()
                ax.set_title('Test # ' + str(test_sample_index + 1) + ' Pred')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_ylabel('z')
                ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
                ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

                ax = fig.add_subplot(num_rows, num_cols, test_sample_index * 3 + 2,
                                     projection='3d')
                z = u_test[test_sample_index]
                plot_3d(xv, yv, zv, z, ax)
                divider = make_axes_locatable(ax)
                ax_cb = divider.new_horizontal(size="5%", pad=0.05)
                # ticks = np.linspace(np.min(z), np.max(z), 8)
                # cbar = fig.colorbar(cs, cax=ax_cb, ticks=ticks)
                # ax_cb.yaxis.tick_right()
                ax.set_title('Test # ' + str(test_sample_index + 1) + ' True')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_ylabel('z')
                ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
                ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

                ax = fig.add_subplot(num_rows, num_cols,test_sample_index * 3 + 3,
                                     projection='3d')
                z = u_test_pred[test_sample_index] - u_test[test_sample_index]
                plot_3d(xv, yv, zv, z, ax)
                divider = make_axes_locatable(ax)
                ax_cb = divider.new_horizontal(size="5%", pad=0.05)
                # ticks = np.linspace(np.min(z),np.max(z),8)
                # cbar = fig.colorbar(cs, cax=ax_cb, ticks=ticks)
                # ax_cb.yaxis.tick_right()
                l2_error = np.sqrt(np.nanmean((u_test_pred[test_sample_index] -
                                               u_test[test_sample_index]) ** 2))
                l2_relative_error = l2_error / np.sqrt(np.nanmean(u_test[test_sample_index] ** 2))
                ax.set_title('L2 Rel. Err. ' + str(l2_relative_error))
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_ylabel('z')
                ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
                ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    l2_error = np.sqrt(np.nanmean((u_test_pred - u_test) ** 2))
    l2_relative_error = l2_error / np.sqrt(np.nanmean((u_test) ** 2))
    fig.suptitle('Test (Epoch ' + str(epoch_index + 1) +
                 ') L2 Rel. Err. ' + str(np.round(l2_relative_error, 4)))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join('debug_figs', 'test',
                             'Fig_Test_epoch' + str(epoch_index + 1) + '.png'))
    plt.close()

    # Loss plots
    plt.figure(1)
    plt.semilogy(np.arange(epoch_index + 1),
                 np.array(loss_train[:epoch_index + 1]), '-b', label='Train L2')
    plt.semilogy(np.arange(epoch_index + 1),
                 np.array(loss_train[:epoch_index + 1]), '-g', label='Train mode')
    plt.semilogy(np.arange(epoch_index + 1),
                 np.array(loss_test[:epoch_index + 1]), '-r', label='Test L2')
    plt.semilogy(np.arange(epoch_index + 1),
                 np.array(loss_test[:epoch_index + 1]), '-c', label='Test mode')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join('debug_figs', 'loss',
                             'Fig_loss_all_epoch' + str(epoch_index + 1) + '.png'))
    plt.close(1)


def save_deeponet(data, u_test_pred, branch_net, trunk_net, bias):
    create_folder('models')
    save_path = os.path.join('models', configs.MODEL_NAME)
    create_folder(save_path)
    x_nodes_test = np.linspace(0.0, 1.0, data['k_train'].shape[1]) # temporary
    model_data = {
        'x_nodes_test': x_nodes_test,
        'k_test': data['k_test'],
        'u_test_true': data['u_test'],
        'u_test_pred': u_test_pred,
    }
    pickle.dump(model_data, open(os.path.join(save_path, 'data_file.pkl'), 'wb'))
    torch.save(branch_net, os.path.join(save_path, 'branch_model'))
    torch.save(trunk_net, os.path.join(save_path, 'trunk_model'))
    torch.save(bias, os.path.join(save_path, 'bias'))


def numerical_derivative(y, dx):
    y_middle = np.concatenate([[y[0]], (y[1:] + y[0:-1]) / 2, [y[-1]]])
    dy_dx = np.diff(y_middle) / dx
    dy_dx[0] = dy_dx[0] / 2
    dy_dx[-1] = dy_dx[-1] / 2
    return dy_dx


def l_shaped_mesh(print_mesh=False):
    if os.path.exists(os.path.join('data', 'L_shaped_50.mat')):
        mat_mesh = loadmat(os.path.join('data', 'L_shaped_50.mat'))
        points = mat_mesh['nodes'] + 0.5
        elements = mat_mesh['elms'] - 1
        # points = 0.1 + np.array([[0, 0], [0, 1], [1, 0], [1, 1],]) / 4
        # elements = np.array([[0, 1, 2], [1, 3, 2]])
    else:
        def get_l_shaped_points(x_lower, x_upper, y_lower, y_upper, n_x, n_y):
            points = np.zeros((n_x * n_y, 2))  # getting chopped later by 'unique'
            points_index = 0
            x_coordinates = np.linspace(x_lower, x_upper, n_x)
            y_coordinates = np.linspace(y_lower, y_upper, n_y)
            rad = 1
            for x_index in range(n_x // 2 + 1):
                for y_index in range(n_y // 2 + 1):
                    if (x_coordinates[x_index] > x_lower and
                        y_coordinates[y_index] > y_lower and
                        x_coordinates[x_index] ** 2 + y_coordinates[y_index] ** 2 > rad ** 2):
                        continue
                    points[points_index, 0] = x_coordinates[x_index]
                    points[points_index, 1] = y_coordinates[y_index]
                    points_index += 1
            for x_index in range(n_x // 2 + 1, n_x):
                for y_index in range(n_y // 2 + 1):
                    if (x_coordinates[x_index] < x_upper and
                        y_coordinates[y_index] > y_lower and
                        x_coordinates[x_index] ** 2 + y_coordinates[y_index] ** 2 > rad ** 2):
                        continue
                    points[points_index, 0] = x_coordinates[x_index]
                    points[points_index, 1] = y_coordinates[y_index]
                    points_index += 1
            for x_index in range(n_x // 2 + 1):
                for y_index in range(n_y // 2 + 1, n_y):
                    if (x_coordinates[x_index] > x_lower and
                        y_coordinates[y_index] < y_upper and
                        x_coordinates[x_index] ** 2 + y_coordinates[y_index] ** 2 > rad ** 2):
                        continue
                    points[points_index, 0] = x_coordinates[x_index]
                    points[points_index, 1] = y_coordinates[y_index]
                    points_index += 1
            return points


        x_lower, x_upper = -1, 1
        y_lower, y_upper = -1, 1
        n_x, n_y = configs.L_SHAPED_N_X, configs.L_SHAPED_N_Y
        levels = configs.L_SHAPED_LEVELS

        points = []
        for _ in range(levels):
            current_level_points = get_l_shaped_points(x_lower, x_upper, y_lower, y_upper, n_x, n_y)
            points.append(current_level_points)
            x_lower, x_upper = 3 * x_lower / 4, 3 * x_upper / 4
            y_lower, y_upper = 3 * y_lower / 4, 3 * y_upper / 4

        points = np.unique(np.vstack(points), axis=0)
        tri_mesh = Delaunay(points)
        elements = tri_mesh.simplices

        mask = [True] * len(elements)
        for simplex_index, simplex in enumerate(elements):
            if (points[simplex[0]][0] >= 0 and points[simplex[0]][1] >= 0 and
                points[simplex[1]][0] >= 0 and points[simplex[1]][1] >= 0 and
                points[simplex[2]][0] >= 0 and points[simplex[2]][1] >= 0):
                mask[simplex_index] = False

        points = (points + 1) / 2

    small_l_bc = np.logical_and(points[:, 0] >= 0.5, points[:, 1] >= 0.5)
    large_l_bc = np.logical_or(points[:, 0] <= 0, points[:, 1] <= 0)
    l_bc = np.logical_or(small_l_bc, large_l_bc)
    influx_bc = points[:, 1] >= 1
    outflux_bc = points[:, 0] >= 1
    l_bc = np.where(influx_bc, False, l_bc)
    l_bc = np.where(outflux_bc, False, l_bc)
    non_bc = np.logical_not(np.logical_or(np.logical_or(l_bc, influx_bc), outflux_bc))

    if print_mesh:
        plt.plot(points[non_bc, 0], points[non_bc, 1], 'o', markersize=1.5)
        plt.plot(points[l_bc, 0], points[l_bc, 1], '*g')
        plt.plot(points[influx_bc, 0], points[influx_bc, 1], '*r')
        plt.plot(points[outflux_bc, 0], points[outflux_bc, 1], '*b')
        plt.triplot(points[:, 0], points[:, 1], elements)
        plt.axis('equal')
        plt.savefig('mesh.png')

    return points, elements, non_bc, l_bc, influx_bc, outflux_bc


def poisson(x_nodes, k_func, f_func):

    dx = x_nodes[1] - x_nodes[0]  # Assuming uniform grid

    # Calculate LHS matrix
    a_mat = np.zeros((len(x_nodes), len(x_nodes)))
    for index in range(1, len(x_nodes) - 1):
        a_mat[index, index] = (k_func[index] + 0.5 * (k_func[index - 1] +
                                                      k_func[index + 1])) / dx ** 2
        a_mat[index, index - 1] = -0.5 * (k_func[index - 1] + k_func[index]) / dx ** 2
        a_mat[index, index + 1] = -0.5 * (k_func[index] + k_func[index + 1]) / dx ** 2
    a_mat[0, 0] = 1
    a_mat[-1, -1] = 1

    # Calculate RHS vector
    loading_vector = f_func

    # Calculate true solution of the (discretized) linear system
    # with zero Dirichlet boundary conditions
    u = np.zeros((len(x_nodes),))
    u[1:-1] = np.linalg.solve(a_mat[1:-1, 1:-1], loading_vector[1:-1])
    du_dx = numerical_derivative(u, dx)

    outputs = {
        'u': u,
        'du_dx': du_dx,
        'a_mat_p': a_mat[1:-1, 1:-1],
        'loading_vector_p': loading_vector[1:-1],
        'non_bc': None
    }

    return outputs


def l_shaped_poisson(x_nodes, y_nodes, k_func, f_func):
    nodes, simplices, non_bc, _, influx_bc, outflux_bc = l_shaped_mesh()

    f_func = np.reshape(f_func, (len(x_nodes), len(y_nodes)))
    xv_transpose, yv_transpose = np.meshgrid(x_nodes, y_nodes)
    k_data_transpose = np.transpose(k_func, (1, 0))
    f_data_transpose = np.transpose(f_func, (1, 0))

    k_func_interp_transpose = interp2d(x_nodes, y_nodes, k_data_transpose)
    f_func_interp_transpose = interp2d(x_nodes, y_nodes, f_data_transpose)
    f_reverse_translator = np.zeros((len(nodes),))

    a_mat = np.zeros((len(nodes), len(nodes)))
    loading_vector = np.zeros((len(nodes),))
    for element in simplices:
        x_1, y_1 = nodes[element[0], 0], nodes[element[0], 1]
        x_2, y_2 = nodes[element[1], 0], nodes[element[1], 1]
        x_3, y_3 = nodes[element[2], 0], nodes[element[2], 1]
        l_1 = np.sqrt((x_2 - x_3) ** 2 + (y_2 - y_3) ** 2)
        l_2 = np.sqrt((x_1 - x_3) ** 2 + (y_1 - y_3) ** 2)
        l_3 = np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)
        s_param = (l_1 + l_2 + l_3) / 2
        area = np.sqrt(s_param * (s_param - l_1) * (s_param - l_2) * (s_param - l_3))
        center_point_x, center_point_y = (x_1 + x_2 + x_3) / 3, (y_1 + y_2 + y_3) / 3
        k_center = k_func_interp_transpose(center_point_x, center_point_y)[0]
        f_center = f_func_interp_transpose(center_point_x, center_point_y)[0]

        a_mat[element[0], element[0]] += (l_1 ** 2) * k_center / (4 * area)
        a_mat[element[1], element[1]] += (l_2 ** 2) * k_center / (4 * area)
        a_mat[element[2], element[2]] += (l_3 ** 2) * k_center / (4 * area)
        a_mat[element[0], element[1]] += - ((l_1 ** 2 + l_2 ** 2 - l_3 ** 2) / 2) * k_center / (4 * area)
        a_mat[element[1], element[0]] += - ((l_1 ** 2 + l_2 ** 2 - l_3 ** 2) / 2) * k_center / (4 * area)
        a_mat[element[0], element[2]] += - ((l_1 ** 2 + l_3 ** 2 - l_2 ** 2) / 2) * k_center / (4 * area)
        a_mat[element[2], element[0]] += - ((l_1 ** 2 + l_3 ** 2 - l_2 ** 2) / 2) * k_center / (4 * area)
        a_mat[element[1], element[2]] += - ((l_2 ** 2 + l_3 ** 2 - l_1 ** 2) / 2) * k_center / (4 * area)
        a_mat[element[2], element[1]] += - ((l_2 ** 2 + l_3 ** 2 - l_1 ** 2) / 2) * k_center / (4 * area)

        loading_vector[element[0]] += f_center * area / 3
        loading_vector[element[1]] += f_center * area / 3
        loading_vector[element[2]] += f_center * area / 3

        f_reverse_translator[element[0]] += area / 3
        f_reverse_translator[element[1]] += area / 3
        f_reverse_translator[element[2]] += area / 3

    a_mat_p = a_mat[non_bc][:, non_bc]
    # loading_vector_p = (loading_vector[non_bc] -
    #                     a_mat[non_bc][:, influx_bc] @ np.ones((np.sum(influx_bc),)) +
    #                     a_mat[non_bc][:, outflux_bc] @ np.ones((np.sum(outflux_bc),)))
    loading_vector_p = loading_vector[non_bc]

    u = pad_dirichlet(np.linalg.solve(a_mat_p, loading_vector_p))
    du_dx = np.zeros_like(u)

    outputs = {
        'u': u,
        'du_dx': du_dx,
        'a_mat_p': a_mat_p,
        'loading_vector_p': loading_vector_p,
        'non_bc': non_bc,
        'f_reverse_translator': f_reverse_translator
    }

    return outputs


def helmholtz(x_nodes, k_func, f_func):
    dx = x_nodes[1] - x_nodes[0]  # Assuming uniform grid

    # Calculate LHS matrix
    a_mat = np.zeros((len(x_nodes), len(x_nodes)))
    for index in range(1, len(x_nodes) - 1):
        a_mat[index, index] = ((dx ** 2) * (k_func[index] ** 2) - 2) / dx ** 2
        a_mat[index, index - 1] = 1 / dx ** 2
        a_mat[index, index + 1] = 1 / dx ** 2
    a_mat[0, 0] = 1
    a_mat[-1, -1] = 1

    # Calculate RHS vector
    loading_vector = f_func
    u = np.zeros((len(x_nodes),))
    u[1:-1] = np.linalg.solve(a_mat[1:-1, 1:-1], loading_vector[1:-1])
    du_dx = numerical_derivative(u, dx)

    outputs = {
        'u': u,
        'du_dx': du_dx,
        'a_mat_p': a_mat[1:-1, 1:-1],
        'loading_vector_p': loading_vector[1:-1],
        'non_bc': None,
    }

    return outputs


def get_non_bc(n_x, n_y, n_z=None):
    if configs.DIMENSIONS == 2:
        non_bc = np.ones(n_x * n_y).astype('bool')
        non_bc[::n_y] = False
        non_bc[n_y-1::n_y] = False
        non_bc[:n_y] = False
        non_bc[-n_y:] = False
    elif configs.DIMENSIONS == 3:
        non_bc = np.ones((n_x, n_y, n_z)).astype('bool')
        non_bc[0, :, :] = False
        non_bc[-1, :, :] = False
        non_bc[:, 0, :] = False
        non_bc[:, -1, :] = False
        non_bc[:, :, 0] = False
        non_bc[:, :, -1] = False
    return non_bc


def helmholtz_2d(x_nodes, y_nodes, k_func, f_func):
    dx = x_nodes[1] - x_nodes[0]  # Assuming uniform grid
    dy = y_nodes[1] - y_nodes[0]  # Assuming uniform grid
    n_x = len(x_nodes)
    n_y = len(y_nodes)
    # Calculate LHS matrix
    # Sequence: (x_0, y_0), (x_0, y_1), ..., (x_0, y_{n_y-1}), (x_1, y_0), ..., (x_{n_x-1}, y_{n_y-1})
    # Can be reshape into (n_x, n_y) rather than (n_y, n_x)
    a_mat = np.eye(n_x * n_y)
    non_bc = get_non_bc(n_x, n_y)

    for index in range(a_mat.shape[0]):
        if non_bc[index]:
            a_mat[index, index] = \
                ((dx ** 2) * (dy ** 2) * (k_func[index // n_y, index % n_y] ** 2) -
                2 * (dx ** 2) - 2 * (dy ** 2))
            a_mat[index, index - 1] = dx ** 2
            a_mat[index, index + 1] = dx ** 2
            a_mat[index, index - n_y] = dy ** 2
            a_mat[index, index + n_y] = dy ** 2

    a_mat = a_mat / ((dx ** 2) * (dy ** 2))

    # Calculate RHS vector
    loading_vector = f_func.flatten()

    a_mat_p = a_mat[non_bc][:, non_bc]
    loading_vector_p = loading_vector[non_bc]

    u = pad_dirichlet(np.linalg.solve(a_mat_p, loading_vector_p))
    du_dx = np.zeros_like(u)

    outputs = {
        'u': u,
        'du_dx': du_dx,
        'a_mat_p': a_mat_p,
        'loading_vector_p': loading_vector_p,
        'non_bc': non_bc,
    }

    return outputs


def helmholtz_3d(x_nodes, y_nodes, z_nodes, k_func, f_func):
    dx = x_nodes[1] - x_nodes[0]  # Assuming uniform grid
    dy = y_nodes[1] - y_nodes[0]  # Assuming uniform grid
    dz = z_nodes[1] - z_nodes[0]  # Assuming uniform grid
    n_x = len(x_nodes)
    n_y = len(y_nodes)
    n_z = len(z_nodes)
    # Calculate LHS matrix
    # Sequence: (x_0, y_0), (x_0, y_1), ..., (x_0, y_{n_y-1}), (x_1, y_0), ..., (x_{n_x-1}, y_{n_y-1})
    # Can be reshape into (n_x, n_y) rather than (n_y, n_x)
    a_mat = np.zeros((n_x, n_y, n_z, n_x, n_y, n_z))
    non_bc = get_non_bc(n_x, n_y, n_z)

    for x_index in range(n_x):
        for y_index in range(n_y):
            for z_index in range(n_z):
                if non_bc[x_index, y_index, z_index]:
                    a_mat[x_index, y_index, z_index,
                          x_index, y_index, z_index] = \
                        ((dx ** 2) * (dy ** 2) * (dz ** 2) *
                         (k_func[x_index, y_index, z_index] ** 2) -
                         2 * ((dx ** 2) * (dy ** 2) + (dx ** 2) * (dz ** 2) +
                         (dy ** 2) * (dz ** 2)))
                    a_mat[x_index, y_index, z_index,
                          x_index - 1, y_index, z_index] = (dy ** 2) * (dz ** 2)
                    a_mat[x_index, y_index, z_index,
                          x_index + 1, y_index, z_index] = (dy ** 2) * (dz ** 2)
                    a_mat[x_index, y_index, z_index,
                          x_index, y_index - 1, z_index] = (dx ** 2) * (dz ** 2)
                    a_mat[x_index, y_index, z_index,
                          x_index, y_index + 1, z_index] = (dx ** 2) * (dz ** 2)
                    a_mat[x_index, y_index, z_index,
                          x_index, y_index, z_index - 1] = (dx ** 2) * (dy ** 2)
                    a_mat[x_index, y_index, z_index,
                          x_index, y_index, z_index + 1] = (dx ** 2) * (dy ** 2)

    a_mat = a_mat / ((dx ** 2) * (dy ** 2) * (dz ** 2))

    non_bc = non_bc.flatten()

    a_mat = np.reshape(a_mat, (n_x * n_y * n_z, n_x * n_y * n_z))
    a_mat_p = a_mat[non_bc][:, non_bc]
    loading_vector_p = f_func.flatten()[non_bc]

    u = pad_dirichlet(np.linalg.solve(a_mat_p, loading_vector_p))
    du_dx = np.zeros_like(u)

    outputs = {
        'u': u,
        'du_dx': du_dx,
        'a_mat_p': a_mat_p,
        'loading_vector_p': loading_vector_p,
        'non_bc': non_bc,
    }

    return outputs


def generate_grf_function(sigma_0, l_0, num_of_elements, num_of_samples):
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


def generate_grf_function_2d(sigma_0, l_0, num_of_elements_x, num_of_elements_y, num_of_samples):
    if len(sigma_0) != len(l_0):
        raise ValueError('Size of sigma_0 and l_0 mismatch')
    num_of_nodes_x = num_of_elements_x + 1
    num_of_nodes_y = num_of_elements_y + 1
    num_of_nodes = num_of_nodes_x * num_of_nodes_y
    x_1d = np.linspace(0, 1, num_of_nodes_x)
    y_1d = np.linspace(0, 1, num_of_nodes_y)
    x, y = np.meshgrid(x_1d, y_1d, indexing='ij')
    x, y = x.flatten(), y.flatten()
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
    return samples.reshape([-1, num_of_nodes_x, num_of_nodes_y])


def plot_3d(x, y, z, c_data, ax=None, slice=False):
    if not slice:
        if ax is None:
            ax = plt.axes(projection='3d')
        ax.grid()
        ax.scatter(x, y, z, c=c_data.flatten(), cmap='viridis')


def generate_grf_function_3d(sigma_0, l_0, num_of_elements_x,
                             num_of_elements_y, num_of_elements_z, num_of_samples):
    if len(sigma_0) != len(l_0):
        raise ValueError('Size of sigma_0 and l_0 mismatch')
    num_of_nodes_x = num_of_elements_x + 1
    num_of_nodes_y = num_of_elements_y + 1
    num_of_nodes_z = num_of_elements_z + 1
    num_of_nodes = num_of_nodes_x * num_of_nodes_y * num_of_nodes_z
    x_1d = np.linspace(0, 1, num_of_nodes_x)
    y_1d = np.linspace(0, 1, num_of_nodes_y)
    z_1d = np.linspace(0, 1, num_of_nodes_z)
    x, y, z = np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    x1, x2 = np.meshgrid(x, x, indexing='ij')
    y1, y2 = np.meshgrid(y, y, indexing='ij')
    z1, z2 = np.meshgrid(z, z, indexing='ij')
    distances_squared  = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    covariance_matrix = np.zeros((num_of_nodes, num_of_nodes))
    for mode_index, corr_length in enumerate(l_0):
        covariance_matrix += ((sigma_0[mode_index] ** 2) *
                               np.exp(- 0.5 / (corr_length ** 2) *
                                      distances_squared))
    mu = np.zeros_like(x)
    samples = np.random.multivariate_normal(mu, covariance_matrix, num_of_samples)

    return samples.reshape([-1, num_of_nodes_x, num_of_nodes_y, num_of_nodes_z])


def get_grf_samples_k(sigma_0=(0.2,), l_0=(0.1,),
                      k_0=1, k_min=0.3,
                      num_of_samples=configs.NUM_OF_CASES):
    redundancy = 6
    num_of_samples_generated = int(np.floor(configs.NUM_OF_CASES * redundancy))
    if configs.DIMENSIONS == 1:
        grf = generate_grf_function(sigma_0, l_0, configs.NUM_OF_ELEMENTS_DON,
                                    num_of_samples_generated) + k_0
        valid_examples = grf[np.where(np.min(grf, axis=1) > k_min)]
    elif configs.DIMENSIONS == 2:
        grf = generate_grf_function_2d(sigma_0, l_0,
                                       configs.NUM_OF_ELEMENTS_DON_X,
                                       configs.NUM_OF_ELEMENTS_DON_Y,
                                       num_of_samples_generated) + k_0
        valid_examples = grf[np.where(np.min(grf, axis=(1, 2)) > k_min)]
    elif configs.DIMENSIONS == 3:
        grf = generate_grf_function_3d(sigma_0, l_0,
                                       configs.NUM_OF_ELEMENTS_DON_X,
                                       configs.NUM_OF_ELEMENTS_DON_Y,
                                       configs.NUM_OF_ELEMENTS_DON_Z,
                                       num_of_samples_generated) + k_0
        valid_examples = grf[np.where(np.min(grf, axis=(1, 2, 3)) > k_min)]

    if configs.L_SHAPED:
        valid_examples[:, valid_examples.shape[1] // 2 + 1:,
                       valid_examples.shape[2] // 2 + 1:] = 0

    if valid_examples.shape[0] < num_of_samples:
        raise ValueError('Need more valid examples')

    return valid_examples[:num_of_samples]


def get_grf_samples_f(sigma_0=(1.0,), l_0=(0.2,),
                      num_of_samples=configs.NUM_OF_CASES):
    if configs.DIMENSIONS == 1:
        grf = generate_grf_function(sigma_0, l_0, configs.NUM_OF_ELEMENTS_DON, num_of_samples)

    elif configs.DIMENSIONS == 2:
        grf = generate_grf_function_2d(sigma_0, l_0, configs.NUM_OF_ELEMENTS_DON_X,
                                       configs.NUM_OF_ELEMENTS_DON_Y, num_of_samples)
    elif configs.DIMENSIONS == 3:
        grf = generate_grf_function_3d(sigma_0, l_0,
                                       configs.NUM_OF_ELEMENTS_DON_X,
                                       configs.NUM_OF_ELEMENTS_DON_Y,
                                       configs.NUM_OF_ELEMENTS_DON_Z, num_of_samples)

    if configs.L_SHAPED:
        grf[:, grf.shape[1] // 2 + 1:, grf.shape[2] // 2 + 1:] = 0
    return grf


def pad_dirichlet(u_p, start=0, end=0):
    if configs.DIMENSIONS == 1:
        u = np.concatenate([[start], u_p, [end]])
    elif configs.DIMENSIONS == 2:
        if configs.L_SHAPED:
            nodes, _, non_bc, _, influx_bc, outflux_bc = l_shaped_mesh()
            u = np.zeros((len(nodes),))
            u[influx_bc] = 0 * np.ones((np.sum(influx_bc),))
            u[outflux_bc] = - 0 * np.ones((np.sum(outflux_bc),))
            u[non_bc] = u_p
        else:
            # TODO: find a way to make generic
            n_x_p = int(np.round(np.sqrt(u_p.shape[0])))
            u = np.hstack(np.pad(np.reshape(u_p, (n_x_p, n_x_p)), 1))
    elif configs.DIMENSIONS == 3:
        # TODO: find a way to make generic
        n_x_p = int(np.round(u_p.shape[0] ** (1 / 3)))
        u = np.pad(np.reshape(u_p, (n_x_p, n_x_p, n_x_p)), 1).flatten()
    return u


def compute_residual(stiff_p, f_p, u_approx, pad=True, non_bc=None):
    if configs.DIMENSIONS == 1:
        residual_p = f_p - stiff_p @ u_approx[1:-1]
    else:
        residual_p = f_p - stiff_p @ u_approx[non_bc]
    if pad:
        return pad_dirichlet(residual_p)
    else:
        return residual_p


def init_simple():
    x_nodes = np.linspace(0.0, 1.0, configs.NUM_OF_ELEMENTS + 1)
    if configs.DIMENSIONS == 1:
        k = 1.5 * configs.K_0 - configs.K_0 * x_nodes
        f = np.ones_like(x_nodes)
        u_init = np.zeros((configs.NUM_OF_ELEMENTS + 1,))
    elif configs.DIMENSIONS == 2:
        x_nodes = np.linspace(0.0, 1.0, configs.NUM_OF_ELEMENTS_X + 1)
        y_nodes = np.linspace(0.0, 1.0, configs.NUM_OF_ELEMENTS_Y + 1)
        if configs.PROBLEM == 'poisson':
            k = np.ones((len(x_nodes), len(y_nodes)))
        elif configs.PROBLEM == 'helmholtz':
            xv, yv = np.meshgrid(x_nodes, y_nodes)
            k = 1.5 * configs.K_0 * np.sin(3 * np.pi * xv) * np.sin(3 * np.pi * yv) * np.exp(xv) * np.exp(yv - 1)
        f = np.ones((len(x_nodes) * len(y_nodes),))
        u_init = np.zeros_like(f)
    elif configs.DIMENSIONS == 3:
        x_nodes = np.linspace(0.0, 1.0, configs.NUM_OF_ELEMENTS_X + 1)
        y_nodes = np.linspace(0.0, 1.0, configs.NUM_OF_ELEMENTS_Y + 1)
        z_nodes = np.linspace(0.0, 1.0, configs.NUM_OF_ELEMENTS_Z + 1)
        if configs.PROBLEM == 'poisson':
            k = np.ones((len(x_nodes), len(y_nodes), len(z_nodes)))
        elif configs.PROBLEM == 'helmholtz':
            k = 2 * np.pi * np.ones((len(x_nodes), len(y_nodes), len(z_nodes)))
        f = np.ones((len(x_nodes), len(y_nodes), len(z_nodes)))
        u_init = np.zeros_like(f).flatten()
    return k, f, u_init
