import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import configs
import utils
from tqdm import tqdm

class DataHandler:
    def __init__(self, logger):
        super(DataHandler, self).__init__()

        self.logger = logger

    def create_data(self):
        np.random.seed(configs.SEED)

        y_nodes = None
        z_nodes = None
        if configs.DIMENSIONS == 1:
            num_of_nodes = configs.NUM_OF_ELEMENTS_DON + 1

            x_nodes = np.linspace(0, 1, num_of_nodes)
            num_of_samples_generated = configs.REDUNDANCY * configs.NUM_OF_CASES
            u_all = np.zeros((num_of_samples_generated, num_of_nodes))
            du_dx_all = np.zeros_like(u_all)
        elif configs.DIMENSIONS == 2:
            n_x = configs.NUM_OF_ELEMENTS_DON_X + 1
            n_y = configs.NUM_OF_ELEMENTS_DON_Y + 1

            x_nodes = np.linspace(0, 1, n_x)
            y_nodes = np.linspace(0, 1, n_y)
            num_of_samples_generated = configs.REDUNDANCY * configs.NUM_OF_CASES
            if configs.L_SHAPED:
                u_all = []
                du_dx_all = []
            else:
                u_all = np.zeros((num_of_samples_generated, n_x, n_y))
                du_dx_all = np.zeros_like(u_all)
        elif configs.DIMENSIONS == 3:
            n_x = configs.NUM_OF_ELEMENTS_DON_X + 1
            n_y = configs.NUM_OF_ELEMENTS_DON_Y + 1
            n_z = configs.NUM_OF_ELEMENTS_DON_Z + 1

            x_nodes = np.linspace(0, 1, n_x)
            y_nodes = np.linspace(0, 1, n_y)
            z_nodes = np.linspace(0, 1, n_z)
            num_of_samples_generated = configs.REDUNDANCY * configs.NUM_OF_CASES
            u_all = np.zeros((num_of_samples_generated, n_x, n_y, n_z))
            du_dx_all = np.zeros_like(u_all)

        if configs.PROBLEM == 'poisson':
            k_data_all = \
                utils.get_grf_samples_k(sigma_0=(configs.K_SIGMA,),
                                        l_0=(configs.K_L0,),
                                        k_0=configs.K_0,
                                        k_min=configs.K_MIN,
                                        num_of_samples=num_of_samples_generated)
        elif configs.PROBLEM == 'helmholtz':
            k_data_all = \
                utils.get_grf_samples_k(sigma_0=(configs.K_SIGMA,),
                                        l_0=(configs.K_L0,),
                                        k_0=configs.K_0,
                                        k_min=configs.K_MIN,
                                        num_of_samples=num_of_samples_generated)
        f_data_all = \
            utils.get_grf_samples_f(sigma_0=(configs.F_SIGMA,),
                                    l_0=(configs.F_L0,),
                                    num_of_samples=num_of_samples_generated)

        for case_index in tqdm(range(num_of_samples_generated)):
            self.logger.debug('Case #%s', str(case_index))
            k_case = k_data_all[case_index]
            f_case = f_data_all[case_index]

            if configs.PROBLEM == 'poisson':
                if configs.L_SHAPED:
                    outputs = \
                        utils.l_shaped_poisson(x_nodes, y_nodes, k_case, f_case)
                else:
                    outputs = \
                        utils.poisson(x_nodes, k_case, f_case)
                u_case = outputs['u']
                du_dx_case = outputs['du_dx']
            elif configs.PROBLEM == 'helmholtz':
                if configs.DIMENSIONS == 1:
                    outputs = utils.helmholtz(x_nodes, k_case, f_case)
                    u_case = outputs['u']
                    du_dx_case = outputs['du_dx']
                elif configs.DIMENSIONS == 2:
                    outputs = utils.helmholtz_2d(x_nodes, y_nodes, k_case, f_case)
                    u_case = np.reshape(outputs['u'], (n_x, n_y))
                    du_dx_case = np.reshape(outputs['du_dx'], (n_x, n_y))
                elif configs.DIMENSIONS == 3:
                    outputs = utils.helmholtz_3d(x_nodes, y_nodes, z_nodes, k_case, f_case)
                    u_case = np.reshape(outputs['u'], (n_x, n_y, n_z))
                    du_dx_case = np.reshape(outputs['du_dx'], (n_x, n_y, n_z))

            if configs.L_SHAPED:
                u_all.append(u_case)
                du_dx_all.append(du_dx_case)
            else:
                u_all[case_index] = u_case
                du_dx_all[case_index] = du_dx_case

        if configs.L_SHAPED:
            u_all = np.stack(u_all)
            du_dx_all = np.stack(du_dx_all)

        # Choose the actual data from *_all according to certain criteria
        if configs.DIMENSIONS == 1:
            u = u_all[:configs.NUM_OF_CASES]
            du_dx = du_dx_all[:configs.NUM_OF_CASES]
            k_data = k_data_all[:configs.NUM_OF_CASES]
            f_data = f_data_all[:configs.NUM_OF_CASES]
        else:
            u_norm_all = np.sqrt(np.mean(u_all.reshape([u_all.shape[0], -1]) ** 2, axis=1))
            valid = u_norm_all < configs.NORM_THRESHOLD
            u, du_dx = u_all[valid], du_dx_all[valid]
            k_data, f_data = k_data_all[valid], f_data_all[valid]
            if u.shape[0] < configs.NUM_OF_CASES:
                raise ValueError('Needs larger value of redundancy for generating data')
            u, du_dx = u[:configs.NUM_OF_CASES], du_dx[:configs.NUM_OF_CASES]
            k_data, f_data = k_data[:configs.NUM_OF_CASES], f_data[:configs.NUM_OF_CASES]

        if configs.SHOW_DATA_CREATION_IMAGES:
            if configs.DIMENSIONS == 1:
                utils.create_folder('debug_figs')
                plt.figure(1,figsize=(6.4, 9.6))
                plt.subplot(311)
                plt.plot(x_nodes, np.transpose(k_data))
                plt.title('GRF k')
                plt.subplot(312)
                plt.plot(x_nodes, np.transpose(f_data))
                plt.title('GRF f')
                plt.subplot(313)
                plt.plot(x_nodes, np.transpose(u))
                plt.title('u')
                plt.tight_layout()
                plt.savefig(os.path.join('debug_figs', 'data.png'))
                plt.close()

        data_preprocessed = {
            'x_nodes': x_nodes,
            'y_nodes': y_nodes,
            'z_nodes': z_nodes,
            'k_data': k_data,
            'f_data': f_data,
            'outputs': np.stack([u, du_dx], axis=1)
        }

        return data_preprocessed

    @staticmethod
    def show_examples(x_nodes, inputs, outputs):
        utils.create_folder('debug_figs')
        num_of_cols = 4
        num_of_plots = 50
        num_of_rows = (num_of_plots - 1) // num_of_cols + 1
        fig1, axs1 = plt.subplots(nrows=num_of_rows, ncols=num_of_cols,
                                figsize=(4.8 * num_of_cols, 3.2 * num_of_rows))
        fig2, axs2 = plt.subplots(nrows=num_of_rows, ncols=num_of_cols,
                                figsize=(4.8 * num_of_cols, 3.2 * num_of_rows))
        for case_index in range(num_of_plots):
            row_index = case_index // num_of_cols
            col_index = case_index % num_of_cols
            ax1 = axs1[row_index, col_index]
            ax2 = axs2[row_index, col_index]
            ax1.plot(x_nodes, inputs[case_index])
            ax2.plot(x_nodes, outputs[case_index, 0])
            ax1.set_title('#' + str(case_index + 1))
            ax2.set_title('#' + str(case_index + 1))
        fig1.tight_layout()
        fig2.tight_layout()
        fig1.savefig(os.path.join('debug_figs', 'data_input.png'))
        fig2.savefig(os.path.join('debug_figs', 'data_output.png'))
        plt.close('all')

    def prepare_data(self, data_preprocessed, percent_test, should_save=True):
        k_data = data_preprocessed['k_data']
        f_data = data_preprocessed['f_data']
        outputs = data_preprocessed['outputs']
        self.logger.debug('Data shape:')
        self.logger.debug('     Input k shape: ' + str(k_data.shape))
        self.logger.debug('     Input f shape: ' + str(f_data.shape))
        self.logger.debug('     Output u shape: ' + str(outputs.shape))

        num_of_samples = k_data.shape[0]

        num_of_test_samples = int(round(num_of_samples * percent_test))
        num_of_train_samples = num_of_samples - num_of_test_samples

        permutated_samples_numbers = np.random.permutation(num_of_samples)
        permutated_k_samples = k_data[permutated_samples_numbers]
        permutated_f_samples = f_data[permutated_samples_numbers]
        permutated_output_samples = outputs[permutated_samples_numbers]

        k_train = permutated_k_samples[:num_of_train_samples]
        f_train = permutated_f_samples[:num_of_train_samples]
        output_train = permutated_output_samples[:num_of_train_samples]
        k_test = permutated_k_samples[num_of_train_samples:]
        f_test = permutated_f_samples[num_of_train_samples:]
        output_test = permutated_output_samples[num_of_train_samples:]

        self.logger.debug('Train-test ratios:')
        self.logger.debug('     Shape of k train samples: ' + str(k_train.shape))
        self.logger.debug('     Shape of f train samples: ' + str(k_train.shape))
        self.logger.debug('     Shape of output train samples: ' + str(output_train.shape))
        self.logger.debug('     Shape of k test samples: ' + str(k_test.shape))
        self.logger.debug('     Shape of f test samples: ' + str(k_test.shape))
        self.logger.debug('     Shape of output test samples: ' + str(output_test.shape))

        u_train = output_train[:, 0]
        du_dx_train = output_train[:, 1]

        u_test = output_test[:, 0]
        du_dx_test = output_test[:, 1]

        data_processed = {
            'x_nodes': data_preprocessed.get('x_nodes'),
            'y_nodes': data_preprocessed.get('y_nodes'),
            'z_nodes': data_preprocessed.get('z_nodes'),
            'k_train': k_train,
            'f_train': f_train,
            'u_train': u_train,
            'du_dx_train': du_dx_train,
            'k_test': k_test,
            'f_test': f_test,
            'u_test': u_test,
            'du_dx_test': du_dx_test,
        }

        if should_save:
            self.save(data_processed)

        return data_processed

    @staticmethod
    def save(data_processed, save_path=None):
        if save_path is None:
            utils.create_folder('data')
            save_path = os.path.join('data', configs.DATASET_NAME)
        pickle.dump(data_processed, open(save_path, 'wb'))

    def get_data(self, percentage_test, data_folder='data', force_create=False):
        data_path = os.path.join(data_folder, configs.DATASET_NAME)
        if os.path.exists(data_path) and not force_create:
            self.logger.info('Loading data from file')
            return pickle.load(open(data_path, 'rb'))
        else:
            self.logger.info('Creating data')
            data_preprocessed = self.create_data()
            return self.prepare_data(data_preprocessed, percentage_test)
