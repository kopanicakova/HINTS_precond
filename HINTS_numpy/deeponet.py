import torch
import torch.nn as nn
import configs
import time
import utils
import numpy as np
from scipy.interpolate import interp2d, griddata, interpn


class DeepONet(nn.Module):
    def __init__(self, x_nodes, logger, y_nodes=None, z_nodes=None):
        super(DeepONet, self).__init__()

        self.logger = logger

        self.x_nodes = x_nodes
        self.y_nodes = y_nodes
        self.z_nodes = z_nodes
        self.generate_trunk_inputs()
        self.relu_activation = nn.ReLU()
        self.tanh_activation = nn.Tanh()
        self.p = 80  # TODO: rename p
        self.num_of_inputs = 2          # k and f herein

        self.num_of_points_x = self.x_nodes.shape[0]
        self.num_of_points_y = self.y_nodes.shape[0] if not y_nodes is None else None
        self.num_of_points_z = self.z_nodes.shape[0] if not z_nodes is None else None

        if configs.DIMENSIONS == 1:
            # Branch Net Compressed Version
            self.branch_net = nn.Sequential(nn.Linear(self.num_of_points_x *
                                                      self.num_of_inputs, 60),
                                            self.relu_activation,
                                            nn.Linear(60, 60),
                                            self.relu_activation,
                                            nn.Linear(60, self.p)).to(configs.DEVICE)
        elif configs.DIMENSIONS == 2:
            num_of_conv_layers = int(np.floor(np.log2(min(len(self.x_nodes), len(self.y_nodes)))))
            layers_list = [nn.Conv2d(in_channels=self.num_of_inputs,
                                                 out_channels=20 + 20,
                                                 kernel_size=(3, 3),
                                                 stride=2,
                                                 padding='valid'),
                           nn.ReLU()]
            for conv_layer_index in range(2, num_of_conv_layers + 1):
                layers_list.append(nn.Conv2d(in_channels=20 * (2 ** (conv_layer_index - 2)) + 20,
                                             out_channels=20 * (2 ** (conv_layer_index - 1)) + 20,
                                             kernel_size=(3, 3),
                                             stride=2,
                                             padding='valid'))
                layers_list.append(nn.ReLU())
            
            layers_list.append(nn.Flatten())
            layers_list.append(nn.Linear(20 * (2 ** (num_of_conv_layers - 1)) + 20, 80))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Linear(80, 80))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Linear(80, self.p))

            self.branch_net = nn.Sequential(*layers_list).to(configs.DEVICE)
        elif configs.DIMENSIONS == 3:
            num_of_conv_layers = int(np.floor(np.log2(min(len(self.x_nodes),
                                                          len(self.y_nodes),
                                                          len(self.z_nodes)))))
            layers_list = [nn.Conv3d(in_channels=self.num_of_inputs,
                                                 out_channels=20 + 20,
                                                 kernel_size=(3, 3, 3),
                                                 stride=2,
                                                 padding='valid'),
                           nn.ReLU()]
            for conv_layer_index in range(2, num_of_conv_layers + 1):
                layers_list.append(nn.Conv3d(in_channels=20 * (2 ** (conv_layer_index - 2)) + 20,
                                             out_channels=20 * (2 ** (conv_layer_index - 1)) + 20,
                                             kernel_size=(3, 3, 3),
                                             stride=2,
                                             padding='valid'))
                layers_list.append(nn.ReLU())
            
            layers_list.append(nn.Flatten())
            layers_list.append(nn.Linear(20 * (2 ** (num_of_conv_layers - 1)) + 20, 80))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Linear(80, 80))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Linear(80, self.p))

            self.branch_net = nn.Sequential(*layers_list).to(configs.DEVICE)

        self.bias = nn.Parameter(torch.tensor(0.0, requires_grad=True,
                                              dtype=torch.float, device=configs.DEVICE))
        # Trunk Net
        self.trunk_net = nn.Sequential(nn.Linear(configs.DIMENSIONS, 80),
                                       self.tanh_activation,
                                       nn.Linear(80, 80),
                                       self.tanh_activation,
                                       nn.Linear(80, self.p),
                                       self.tanh_activation).to(configs.DEVICE)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=configs.LR)

        # Learning rate scheduler
        decay_rate = 0.5 ** (1 / 1000)
        self.scheduler = \
            torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate)

    def _forward_branch(self, inputs):
        return self.branch_net(inputs)

    def generate_trunk_inputs(self, inputs=None):
        if inputs is None:
            if configs.DIMENSIONS == 1:
                self.trunk_inputs = torch.tensor(self.x_nodes[:, None],
                                                 requires_grad=True,
                                                 dtype=torch.float,
                                                 device=configs.DEVICE)
            elif configs.DIMENSIONS == 2:
                if configs.L_SHAPED:
                    points, _, _, _, _, _ = utils.l_shaped_mesh()
                    self.trunk_inputs = \
                        torch.tensor(points, requires_grad=True,
                                     dtype=torch.float, device=configs.DEVICE)                   
                else:
                    xv, yv = np.meshgrid(self.x_nodes, self.y_nodes, indexing='ij')
                    xv_flat, yv_flat = xv.flatten(), yv.flatten()
                    self.trunk_inputs = \
                        torch.tensor(np.stack([xv_flat, yv_flat], axis=1),
                                     requires_grad=True,
                                     dtype=torch.float, device=configs.DEVICE)
            elif configs.DIMENSIONS == 3:
                xv, yv, zv = np.meshgrid(self.x_nodes, self.y_nodes,
                                            self.z_nodes, indexing='ij')
                xv_flat = xv.flatten()
                yv_flat = yv.flatten()
                zv_flat = zv.flatten()
                self.trunk_inputs = \
                    torch.tensor(np.stack([xv_flat, yv_flat, zv_flat], axis=1),
                                    requires_grad=True,
                                dtype=torch.float, device=configs.DEVICE)
        else:
            return torch.tensor(inputs[:, None], requires_grad=True,
                                dtype=torch.float, device=configs.DEVICE)

    def _forward_trunk(self, inputs=None):
        if inputs is None:
            return self.trunk_net(self.trunk_inputs)
        else:
            return self.trunk_net(inputs)

    def forward(self, branch_input_k, branch_input_f, trunk_inputs=None):
        # branch input (k or f) shape: (batch size, num_of_points)
        if trunk_inputs is None:
            trunk_inputs = self.trunk_inputs
            is_internal = True
        else:
            trunk_inputs = self.generate_trunk_inputs(trunk_inputs)
            is_internal = False
        trunk_outputs = self._forward_trunk(trunk_inputs)       # (num_points, self.p)

        # k(x): normalize [0, 2] --> [-1, 1]
        if configs.PROBLEM == 'poisson':
            branch_input_k = (branch_input_k - configs.K_0) / configs.K_SIGMA
        elif configs.PROBLEM == 'helmholtz':
            branch_input_k = (branch_input_k - configs.K_0) / configs.K_SIGMA

        if configs.DIMENSIONS == 1:
            # f(x), or the RHS: make L2 norm = 1, and save the magnitude (multiply back later)
            #f_norm = torch.linalg.vector_norm(branch_input_f, dim=1, keepdim=True)
            f_norm = torch.sqrt(torch.mean(branch_input_f**2,dim=1,keepdim=True))
            branch_input_f = branch_input_f / f_norm

            branch_inputs = torch.cat((branch_input_k, branch_input_f), dim=1)
            branch_outputs = self._forward_branch(branch_inputs)    # (num_cases, self.p)

            u_pred = branch_outputs @ trunk_outputs.T       # (num_cases, num_points)
            u_pred = u_pred * f_norm * trunk_inputs.T * (trunk_inputs.T - 1.0)
            du_dx_pred = u_pred  # disable the function of dudx, but keep the codes structure for now
        elif configs.DIMENSIONS == 2:
            # f(x), or the RHS: make L2 norm = 1, and save the magnitude (multiply back later)
            f_norm = torch.sqrt(torch.mean(branch_input_f ** 2, dim=(1, 2), keepdim=True))
            branch_input_f = branch_input_f / f_norm

            branch_inputs = torch.cat((branch_input_k[:, None], branch_input_f[:, None]), dim=1)
            branch_outputs = self._forward_branch(branch_inputs)

            # When training
            if configs.L_SHAPED:
                u_pred = branch_outputs @ trunk_outputs.T[:, 0, :]
                u_pred = u_pred * f_norm[:, 0]
                du_dx_pred = u_pred  # disable the function of dudx, but keep the codes structure for now
            else:
                if is_internal:
                    u_pred = torch.reshape(branch_outputs @ trunk_outputs.T,
                                           (branch_outputs.shape[0],
                                            len(self.x_nodes),
                                            len(self.y_nodes)))
                    xv, yv = torch.meshgrid(torch.tensor(self.x_nodes, device=configs.DEVICE),
                                            torch.tensor(self.y_nodes, device=configs.DEVICE), indexing='ij')
                else: # When inferring
                    # TODO: not use sqrt for reshaping
                    x_nodes = np.linspace(0.0, 1.0, int(np.sqrt(trunk_inputs.shape[0])))
                    y_nodes = np.linspace(0.0, 1.0, int(np.sqrt(trunk_inputs.shape[0])))
                    u_pred = torch.reshape(branch_outputs @ trunk_outputs.T[:, 0, :],
                                        (branch_outputs.shape[0],
                                            len(x_nodes),
                                            len(y_nodes)))
                    xv, yv = torch.meshgrid(torch.tensor(x_nodes, device=configs.DEVICE),
                                            torch.tensor(y_nodes, device=configs.DEVICE), indexing='ij')
                u_pred = u_pred * f_norm
            if configs.L_SHAPED:
                if configs.LIFTING:
                    theta = torch.atan2(0.5 - self.trunk_inputs[:, 1] + 10 ** (-8),
                                        0.5 - self.trunk_inputs[:, 0] + 10 ** (-8))
                    phy = - (theta + np.pi / 2) * (theta - np.pi)
                    u_pred = (u_pred * self.trunk_inputs[:, 0] * (self.trunk_inputs[:, 0] - 1) *
                            self.trunk_inputs[:, 1] * (self.trunk_inputs[:, 0] - 1) * phy *
                            torch.sqrt((self.trunk_inputs[:, 0] - 0.5) ** 2 + 
                                    (self.trunk_inputs[:, 1] - 0.5) ** 2))
            else:
                u_pred = u_pred * xv[None, :, :] * (xv[None, :, :] - 1.0)
                u_pred = u_pred * yv[None, :, :] * (yv[None, :, :] - 1.0)
            du_dx_pred = u_pred  # disable the function of dudx, but keep the codes structure for now
        elif configs.DIMENSIONS == 3:
            # f(x), or the RHS: make L2 norm = 1, and save the magnitude (multiply back later)
            f_norm = torch.sqrt(torch.mean(branch_input_f ** 2, dim=(1, 2, 3), keepdim=True))
            branch_input_f = branch_input_f / f_norm

            branch_inputs = torch.cat((branch_input_k[:, None], branch_input_f[:, None]), dim=1)
            branch_outputs = self._forward_branch(branch_inputs)

            # When training
            if is_internal:
                u_pred = torch.reshape(branch_outputs @ trunk_outputs.T,
                                       (branch_outputs.shape[0],
                                        len(self.x_nodes),
                                        len(self.y_nodes),
                                        len(self.z_nodes)))
                xv, yv, zv = torch.meshgrid(torch.tensor(self.x_nodes, device=configs.DEVICE),
                                            torch.tensor(self.y_nodes, device=configs.DEVICE),
                                            torch.tensor(self.z_nodes, device=configs.DEVICE),
                                            indexing='ij')
            else: # When inferring
                # TODO: not use sqrt for reshaping
                x_nodes = np.linspace(0.0, 1.0, int(np.round(trunk_inputs.shape[0] ** (1 / 3))))
                y_nodes = np.linspace(0.0, 1.0, int(np.round(trunk_inputs.shape[0] ** (1 / 3))))
                z_nodes = np.linspace(0.0, 1.0, int(np.round(trunk_inputs.shape[0] ** (1 / 3))))
                u_pred = torch.reshape(branch_outputs[0] @ trunk_outputs[:, 0, :].T,
                                       (branch_outputs.shape[0],
                                        len(x_nodes),
                                        len(y_nodes),
                                        len(z_nodes)))
                xv, yv, zv = torch.meshgrid(torch.tensor(x_nodes, device=configs.DEVICE),
                                            torch.tensor(y_nodes, device=configs.DEVICE),
                                            torch.tensor(z_nodes, device=configs.DEVICE),
                                            indexing='ij')
            u_pred = u_pred * f_norm
            u_pred = u_pred * xv[None, :, :] * (xv[None, :, :] - 1.0)
            u_pred = u_pred * yv[None, :, :] * (yv[None, :, :] - 1.0)
            u_pred = u_pred * zv[None, :, :] * (zv[None, :, :] - 1.0)
            du_dx_pred = u_pred  # disable the function of dudx, but keep the codes structure for now

        return u_pred, du_dx_pred

    @staticmethod
    def losses(u_true, u_pred, du_dx_true, du_dx_pred):
        if configs.LOSS_FUNC == 'L2_abs':
            return (torch.mean((u_pred - u_true) ** 2) +
                              0.0 * torch.mean((du_dx_pred - du_dx_true) ** 2))
        elif configs.LOSS_FUNC == 'L2_relative':
            return (torch.mean((u_pred - u_true) ** 2 / (u_true ** 2 + 1E-2)) +
                    0.0 * torch.mean((du_dx_pred - du_dx_true) ** 2))
        elif configs.LOSS_FUNC == 'L2_between':
            return (torch.mean((u_pred - u_true) ** 2 / (torch.abs(u_true) + 1E-3)) +
                    0.0 * torch.mean((du_dx_pred - du_dx_true) ** 2))
        elif configs.LOSS_FUNC == 'L2_log':
            return (torch.mean(torch.abs(u_pred - u_true) *
                    torch.log(1.0 + 10 * torch.abs(u_pred - u_true)) / 10) +
                    0.0 * torch.mean((du_dx_pred - du_dx_true) ** 2))
        else:
            raise ValueError('LOSS_FUNC not defined')

    def fit(self, data, num_of_epochs=5000, batch_size=100,
            plot_interval=0, should_save=None):
        utils.create_folder('debug_figs')

        k_train = data['k_train']
        f_train = data['f_train']
        u_train = data['u_train']
        du_dx_train = data['du_dx_train']

        k_test = data['k_test']
        f_test = data['f_test']
        u_test = data['u_test']
        du_dx_test = data['du_dx_test']

        num_of_samples_train = k_train.shape[0]
        batches, num_of_batches = utils.batch_loader(num_of_samples_train, batch_size)

        num_of_samples_test = k_test.shape[0]
        x_nodes_test = np.linspace(0.0, 1.0, data['k_train'].shape[1])  # for now, the same

        loss_train = np.zeros((num_of_epochs))
        loss_test = np.zeros_like(loss_train)
        samples_permutation = np.random.permutation(num_of_samples_train)
        for epoch_index in range(num_of_epochs):
            tic = time.time()
            loss_sum = 0
            for batch_index in range(num_of_batches):
                start_index = batches[batch_index, 0]
                end_index = batches[batch_index, 1]
                k_batch = k_train[samples_permutation[start_index:end_index]]
                f_batch = f_train[samples_permutation[start_index:end_index]]
                u_batch_pred, du_dx_batch_pred = self.forward(utils.to_torch(k_batch),
                                                              utils.to_torch(f_batch))
                u_batch = u_train[samples_permutation[start_index:end_index]]
                du_dx_batch = du_dx_train[samples_permutation[start_index:end_index]]
                # TODO: totorch at the beginning
                loss_batch = \
                    self.losses(utils.to_torch(u_batch), u_batch_pred,
                                utils.to_torch(du_dx_batch), du_dx_batch_pred)
                loss_sum += utils.to_numpy(loss_batch) * batches[batch_index, 2]
                self.optimizer.zero_grad()
                loss_batch.backward()
                self.optimizer.step()
                # if epoch_index in []:
                #    self.scheduler.step()

            train_loss_epoch = loss_sum / num_of_samples_train
            loss_train[epoch_index] = train_loss_epoch
            toc = time.time()
            self.logger.debug('Epoch: ' + str(epoch_index + 1) +
                              ', Loss (x1E-4): (' + str(np.round(train_loss_epoch * 1E4, 4)) +
                              '), Time: ' + str(int((toc - tic) * 100)) + 's')
            if epoch_index % configs.PRINT_INTERVAL == 0:
                self.logger.info('Epoch: ' + str(epoch_index + 1) +
                                 ', Loss (x1E-4): (' + str(np.round(train_loss_epoch * 1E4, 4)) +
                                 '), Time: ' + str(int((toc - tic) * 100)) + 's')

            u_test_pred, du_dx_test_pred = self.forward(utils.to_torch(k_test),
                                                        utils.to_torch(f_test))
            test_loss_epoch = self.losses(utils.to_torch(u_test), u_test_pred,
                                          utils.to_torch(du_dx_test), du_dx_test_pred)
            loss_test[epoch_index] = utils.to_numpy(test_loss_epoch)

            if plot_interval > 0 and (epoch_index + 1) % plot_interval == 0:
                # TODO: pack (too many variables)
                u_test_pred = utils.to_numpy(u_test_pred)
                utils.results_plotter(epoch_index, num_of_samples_test,
                                      x_nodes_test, u_test_pred, u_test,
                                      loss_train, loss_test)

        if should_save:
            utils.save_deeponet(data, u_test_pred, self.branch_net,
                                self.trunk_net, self.bias)

    def infer(self, k_data, f_data, save_path=None):
        # TODO: Change the hard-coded [0,1] interval?
        self.logger.debug('DeepONet infer')
        if configs.DIMENSIONS == 1:
            if k_data.shape[1] != self.num_of_points_x:
                x_data = np.linspace(0, 1, k_data.shape[1])
                k_data_interp = np.interp(self.x_nodes, x_data, k_data[0])[None,:]
                f_data_interp = np.interp(self.x_nodes, x_data, f_data[0])[None,:]
                torch_k_data = utils.to_torch(k_data_interp)
                torch_f_data = utils.to_torch(f_data_interp)
            else:
                torch_k_data = utils.to_torch(k_data)
                torch_f_data = utils.to_torch(f_data)
            model_output = self.forward(torch_k_data, torch_f_data,
                                        np.linspace(0.0, 1.0,
                                        k_data.shape[1]))
        elif configs.DIMENSIONS == 2:
            x_data = np.linspace(0, 1, k_data.shape[1])
            y_data = np.linspace(0, 1, k_data.shape[2])
            xv_transpose, yv_transpose = np.meshgrid(x_data, y_data)
            if k_data.shape[1] != self.num_of_points_x or k_data.shape[2] != self.num_of_points_y:
                if configs.L_SHAPED:
                    points, _, _, _, _, _ = utils.l_shaped_mesh()
                    f_data_transpose = \
                        griddata(points, f_data[0], (xv_transpose,
                                                     yv_transpose), method='cubic')[None, :]
                    f_data = np.nan_to_num(np.transpose(f_data_transpose, (0, 2, 1)))
                else:
                    points = np.concatenate((xv_transpose.transpose().reshape([-1, 1]),
                                             yv_transpose.transpose().reshape([-1, 1])), axis=1)
                    f_data = np.reshape(f_data, (f_data.shape[0], k_data.shape[1], k_data.shape[2]))

                # Only in this function, adopt xy indexing (denoted as ji herein) to be consistent with interp2d
                k_data_transpose = np.transpose(k_data, (0, 2, 1))
                f_data_transpose = np.transpose(f_data, (0, 2, 1))

                k_func_interp_transpose = interp2d(x_data, y_data, k_data_transpose[0])
                f_func_interp_transpose = interp2d(x_data, y_data, f_data_transpose[0])

                k_data_interp_transpose = \
                    k_func_interp_transpose(self.x_nodes, self.y_nodes)[None, :]
                f_data_interp_transpose = \
                    f_func_interp_transpose(self.x_nodes, self.y_nodes)[None, :]

                k_data_interp = np.transpose(k_data_interp_transpose, (0, 2, 1))
                f_data_interp = np.transpose(f_data_interp_transpose, (0, 2, 1))

                torch_k_data = utils.to_torch(k_data_interp)
                torch_f_data = utils.to_torch(f_data_interp)
            else:
                if configs.L_SHAPED:
                    points, _, _, _, _, _ = utils.l_shaped_mesh()
                    f_data_transpose = \
                        griddata(points, f_data[0], (xv_transpose,
                                                     yv_transpose), method='cubic')[None, :]
                    f_data = np.nan_to_num(np.transpose(f_data_transpose, (0, 2, 1)))
                torch_k_data = utils.to_torch(k_data)
                torch_f_data = utils.to_torch(f_data)


            model_output = self.forward(torch_k_data, torch_f_data, points)
        elif configs.DIMENSIONS == 3:
            x_data = np.linspace(0, 1, k_data.shape[1])
            y_data = np.linspace(0, 1, k_data.shape[2])
            z_data = np.linspace(0, 1, k_data.shape[3])
            xv_transpose, yv_transpose, zv_transpose = np.meshgrid(x_data,
                                                                   y_data,
                                                                   z_data)
            if (k_data.shape[1] != self.num_of_points_x or
                k_data.shape[2] != self.num_of_points_y or
                k_data.shape[3] != self.num_of_points_z):

                xv_deeponet, yv_deeponet, zv_deeponet = np.meshgrid(self.x_nodes,
                                                                    self.y_nodes,
                                                                    self.z_nodes)
                deeponet_points = np.concatenate((xv_deeponet.transpose().reshape([-1, 1]),
                                                  yv_deeponet.transpose().reshape([-1, 1]),
                                                  zv_deeponet.transpose().reshape([-1, 1])), axis=1)

                points = np.concatenate((xv_transpose.transpose().reshape([-1, 1]),
                                         yv_transpose.transpose().reshape([-1, 1]),
                                         zv_transpose.transpose().reshape([-1, 1])), axis=1)
                f_data = np.reshape(f_data, (f_data.shape[0], k_data.shape[1],
                                             k_data.shape[2], k_data.shape[3]))

                # Only in this function, adopt xy indexing (denoted as ji herein) to be consistent with interp2d
                k_data_transpose = np.transpose(k_data, (0, 3, 2, 1))
                f_data_transpose = np.transpose(f_data, (0, 3, 2, 1))

                k_func_transpose_vec = interpn((x_data, y_data, z_data),
                                               k_data_transpose[0], deeponet_points)
                f_func_transpose_vec = interpn((x_data, y_data, z_data),
                                               f_data_transpose[0], deeponet_points)
                
                k_func_transpose = np.reshape(k_func_transpose_vec, (1, self.num_of_points_x,
                                                                     self.num_of_points_y,
                                                                     self.num_of_points_z))
                f_func_transpose = np.reshape(f_func_transpose_vec, (1, self.num_of_points_x,
                                                                     self.num_of_points_y,
                                                                     self.num_of_points_z))

                k_data_interp = np.transpose(k_func_transpose, (0, 3, 2, 1))
                f_data_interp = np.transpose(f_func_transpose, (0, 3, 2, 1))

                torch_k_data = utils.to_torch(k_data_interp)
                torch_f_data = utils.to_torch(f_data_interp)
            else:
                torch_k_data = utils.to_torch(k_data)
                torch_f_data = utils.to_torch(f_data)

            model_output = self.forward(torch_k_data, torch_f_data, points)

        u_pred = model_output[0].cpu().detach().numpy()
        du_dx_pred = model_output[1].cpu().detach().numpy()

        if save_path is not None:
            np.savez(save_path, u_pred=u_pred, du_dx_pred=du_dx_pred)

        return u_pred, du_dx_pred
