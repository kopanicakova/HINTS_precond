import matplotlib.pyplot as plt
import os, shutil
import torch
import torch.nn as nn
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import *
import subprocess
import copy
import sys

from utils  import create_folder, to_numpy, cleanup_folder
from config import params as args
from config import DEVICE
from utils import logger 


class DistributedBatchSampler(torch.utils.data.sampler.RandomSampler):
    def __init__(self, batch_sampler, **kwargs):
        self.batch_sampler = batch_sampler
        self.kwargs = kwargs

    def __iter__(self):
        for batch in self.batch_sampler:
            yield list(DistributedSampler(batch, **self.kwargs))

    def __len__(self):
        return len(self.batch_sampler)



class Trainer(object):
    def __init__(self):
        self.non_saving_count = 0


    def setup_optimizer(self, net):
        # Optimizer
        if(args.optimizer_type == "lbfgs"):
            logger.info("Using LBFGS optimizer, lr: "+str(args.lr))
            # self.optimizer = torch.optim.LBFGS(net.parameters(), lr=args.lr, history_size=3, max_iter=1, line_search_fn="strong_wolfe")
            self.optimizer = torch.optim.LBFGS(net.parameters(), lr=args.lr, history_size=3, max_iter=1)
            # self.optimizer = LBFGSMomentum(net.parameters())
            
        else:
            logger.info("Using Adam optimizer, lr: "+str(args.lr))
            # self.optimizer = torch.optim.Adam(net.parameters(), lr=configs.LR, weight_decay=1e-5)
            self.optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
            
            # Learning rate scheduler
            decay_rate = 0.5 ** (1 / 1000)
            self.scheduler = \
                torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate)




    def setup_data_loaders(self, net, data, batch_size):

        if('features_train' in data):
            self.non_nested_mesh = False
            dataset_train = net.dataset_type(   data_u=data['u_train'],
                                                data_features=data['features_train'],
                                                permutation_indices=net.permutation_indices,
                                                reshape_feature_tensor=net.reshape_feature_tensor)

            dataset_test = net.dataset_type(    data_u=data['u_test'],
                                                data_features=data['features_test'],
                                                permutation_indices=net.permutation_indices,
                                                reshape_feature_tensor=net.reshape_feature_tensor)

        # Non-nested meshes 
        else:
            dataset_train = net.dataset_type(   data_u=data['u_train'],
                                                data_features=data['features_don_train'],
                                                permutation_indices=net.permutation_indices,
                                                reshape_feature_tensor=net.reshape_feature_tensor)

            dataset_test = net.dataset_type(    data_u=data['u_test'],
                                                data_features=data['features_don_test'],
                                                permutation_indices=net.permutation_indices,
                                                reshape_feature_tensor=net.reshape_feature_tensor)
            self.non_nested_mesh = True

        bs = batch_size


        drop_last_train = True if len(dataset_train) > bs else False
        drop_last_test  = True if len(dataset_test) > bs else False


        sampler_train = torch.utils.data.sampler.BatchSampler(  torch.utils.data.sampler.RandomSampler(dataset_train),
                                                                batch_size=bs,
                                                                drop_last=drop_last_train)


        sampler_test = torch.utils.data.sampler.BatchSampler(   torch.utils.data.sampler.RandomSampler(dataset_test),
                                                                batch_size=bs,
                                                                drop_last=drop_last_test)


        if dist.is_initialized() and dist.get_world_size()>1:
            sampler_train = DistributedBatchSampler(sampler_train, num_replicas=dist.get_world_size(), rank=dist.get_rank())
            sampler_test  = DistributedBatchSampler(sampler_test, num_replicas=dist.get_world_size(), rank=dist.get_rank())


        data_loader_train = DataLoader( dataset_train,
                                        # pin_memory=True,
                                        # num_workers=5,
                                        # shuffle=True,
                                        sampler=sampler_train)

        data_loader_test = DataLoader(  dataset_test,
                                        # pin_memory=True,
                                        # num_workers=5,
                                        # shuffle=True,
                                        sampler=sampler_test)


        return  data_loader_train, data_loader_test



    @staticmethod
    def save_model(net, name):
        create_folder('models')
        save_path = os.path.join('models', name)

        if dist.is_initialized() and dist.get_world_size()>1:
            torch.save(net.module.state_dict(), save_path)
        else:
            torch.save(net.state_dict(), save_path)



    # @staticmethod
    def load_model(self, net):
        if(args.model_name_load=="none"):
            PATH = os.path.join('models', args.model_name)
            logger.info("load_model:: PATH " + str(PATH))
        else:
            PATH = os.path.join('models', args.model_name_load)
            logger.info("load_model:: PATH " + str(PATH))

        if(os.path.isfile(PATH)):
            net.load_state_dict(torch.load(PATH, map_location=DEVICE))
            return True
        else:
            logger.info("DeepOnet not found ... ")
            return False
    

    # TO DO:: fix
    # @staticmethod
    def losses(self, u_true, u_pred):

        if args.loss_function == 'L2_abs':
            loss =  (torch.mean((u_pred - u_true) ** 2))
        elif args.loss_function == 'L2_relative':
            if(u_true.dtype == torch.complex64):
                loss_real   =  (torch.mean((u_pred.real - u_true.real) ** 2 / (u_true.real ** 2 + 1E-4)))
                loss_img    =  (torch.mean((u_pred.imag - u_true.imag) ** 2 / (u_true.imag ** 2 + 1E-4)))
                loss        = loss_real + loss_img

            else:
                loss =  (torch.mean((u_pred - u_true) ** 2 / (u_true ** 2 + 1E-4)))

        elif args.loss_function == 'L2_between':
            loss =  (torch.mean((u_pred - u_true) ** 2 / (torch.abs(u_true) + 1E-3)))

        elif args.loss_function == 'L2_log':
            loss =  (torch.mean(torch.abs(u_pred - u_true) *
                    torch.log(1.0 + 10 * torch.abs(u_pred - u_true)) / 10))

        elif (args.loss_function == 'MSE' or args.loss_function=="pinn"):
            loss =  torch.mean(((u_pred) ** 2))
            # Relative MSE
            # loss =  torch.mean(((u_pred) ** 2)/u_pred)   
        else:
            raise ValueError('LOSS_FUNC not defined')

        if dist.is_initialized() and dist.get_world_size()>1:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss_sum_reduced = 1./dist.get_world_size()*loss

        else:
            loss_sum_reduced = loss


        return loss_sum_reduced



    def plot_train_results(self, epoch, dop, features_all, net_out_all, targets_all):

        test_cases  = 10
        num_rows    = 10

        if(self.non_nested_mesh):
            num_cols    = 2
        else:
            num_cols    = 3

        fig, axs    = plt.subplots(nrows=num_rows, ncols=num_cols,
                                figsize=(3.1 * num_cols, 2.4 * num_rows))
        fig.tight_layout()

        samples     = np.random.randint(0, high=10, size=num_rows, dtype=int)

        net_out_all = net_out_all.cpu().detach().numpy()


        if(len(net_out_all.shape)==3):
            net_out_all = net_out_all[0]


        for plot_id, sample_id in enumerate(samples):

            y_np  = net_out_all[sample_id]

            if(self.non_nested_mesh==False):
                exact = targets_all[0][sample_id]
                exact = exact.cpu().detach().numpy()


            np.set_printoptions(threshold=sys.maxsize)

            if(self.non_nested_mesh==False):
                error = np.abs(y_np-exact)


            ax = axs[plot_id, 0]
            if(dop.reshape_feature_tensor is not None):
                contours = ax.contour(np.reshape(y_np, (dop.reshape_feature_tensor[1],dop.reshape_feature_tensor[2])))
                ax.set_aspect("equal")
                plt.colorbar(contours, ax=ax)                
            else:
                contours = ax.plot(y_np)
            ax.set_title("Net") 

            if(self.non_nested_mesh==False):
                ax = axs[plot_id, 1]
                if(dop.reshape_feature_tensor is not None):
                    contours = ax.contour(np.reshape(exact, (dop.reshape_feature_tensor[1],dop.reshape_feature_tensor[2])))
                    ax.set_aspect("equal")
                    plt.colorbar(contours, ax=ax)                
                else:
                    contours = ax.plot(exact)            
                ax.set_title("Exact") 

                
                ax = axs[plot_id, 2]
                if(dop.reshape_feature_tensor is not None):
                    contours = ax.contour(np.reshape(error, (dop.reshape_feature_tensor[1],dop.reshape_feature_tensor[2])))
                    ax.set_aspect("equal")
                    plt.colorbar(contours, ax=ax)                
                else:
                    contours = ax.plot(error)       
                ax.set_title("Error") 



        plt.savefig("debug_figs/test"+str(args.model_name)+'_'+str(args.dataset_name)+"/test_new_"+str(epoch)+".png")
        plt.close()

    

    def fit(self, net, data, num_of_epochs=5000, batch_size=100, plot_interval=0, should_save=None):

        if ( dist.is_initialized()==False or (dist.is_initialized() and dist.get_rank()==0)):
            logger.info("Training  initiated.")
            create_folder(os.path.join('debug_figs'))
            create_folder(os.path.join('debug_figs/', 'test'+str(args.model_name)+'_'+str(args.dataset_name)))
            create_folder(os.path.join('debug_figs/', 'loss'+str(args.model_name)+'_'+str(args.dataset_name)))


            cleanup_folder('debug_figs/loss'+str(args.model_name)+'_'+str(args.dataset_name))
            cleanup_folder('debug_figs/test'+str(args.model_name)+'_'+str(args.dataset_name))
                       

        data_loader_train, data_loader_test = self.setup_data_loaders(net, data, batch_size)
        best_test_loss = 9e9

        if dist.is_initialized() and dist.get_world_size()>1:
            # dop = DistributedDataParallel(net, device_ids=range(torch.cuda.device_count()), find_unused_parameters=True)
            dop = DistributedDataParallel(net, device_ids=range(torch.cuda.device_count()))
            # dop.module.init_params()
        else:
            dop=net
            # dop.init_params()


        self.setup_optimizer(dop)

        num_of_epochs = num_of_epochs+1


        loss_train = np.zeros((num_of_epochs))
        loss_test = 9e9*np.ones_like(loss_train)
        dop.train()

        ratio=args.max_stall/args.print_interval
        logger.info("max_stall_local " + str(ratio))


        for epoch_index in range(0, num_of_epochs):

            if(self.non_saving_count > args.max_stall/args.print_interval):
                break


            loss_sum = 0
            tic = time.time()
            for batch_idx, (features, targets) in enumerate(data_loader_train):

                def closure():
                    if torch.is_grad_enabled():
                        self.optimizer.zero_grad()    

                    if dist.is_initialized() and dist.get_world_size()>1:
                        u_batch_pred = dop.forward(features, dop.module.trunk_inputs)
                    else:
                        u_batch_pred = dop.forward(features, dop.trunk_inputs)

                        
                    if(len(u_batch_pred.shape)==3):
                        u_batch_pred = u_batch_pred[0]

                    loss_batch = self.losses(targets[0], u_batch_pred)

                    if((epoch_index % plot_interval == 0) and batch_idx==0):
                        if ( dist.is_initialized()==False or (dist.is_initialized() and dist.get_rank()==0)):

                            if(args.loss_function=="pinn"):
                                if dist.is_initialized() and dist.get_world_size()>1: # check that logic
                                    u_batch_pred = dop.don.forward(features[0], dop.module.trunk_inputs)
                                else:
                                    u_batch_pred = dop.don.forward(features[0], dop.trunk_inputs)                                
                                # self.plot_train_results(epoch_index, dop, features, u_batch_pred, targets)
                            else:
                                pass
                                # self.plot_train_results(epoch_index, dop, features, u_batch_pred, targets)
                                
                                # if(self.non_nested_mesh):
                                #     u_batch_pred_plot = dop.forward(features, dop.trunk_inputs_don)
                                #     self.plot_train_results(epoch_index, dop, features, u_batch_pred_plot, targets)
                                # else:
                                #     self.plot_train_results(epoch_index, dop, features, u_batch_pred, targets)




                    if loss_batch.requires_grad:
                        loss_batch.backward()

                    
                    return loss_batch


                loss_batch = self.optimizer.step(closure)
                loss_sum += loss_batch



            if dist.is_initialized() and dist.get_world_size()>1:
                with torch.no_grad():
                    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                    loss_sum_reduced = 1./dist.get_world_size()*loss_sum
                    loss_sum_reduced = to_numpy(loss_sum_reduced)
            else:
                loss_sum_reduced = to_numpy(loss_sum)



            loss_train[epoch_index] = loss_sum_reduced/(batch_idx+1)

            toc = time.time()
            if epoch_index % args.print_interval == 0:
                if ( dist.is_initialized()==False or (dist.is_initialized() and dist.get_rank()==0)):
                    logger.info('Epoch: ' + str(epoch_index + 1) +
                                     ', Loss (x1E-4): (' + str(np.round(loss_train[epoch_index] * 1E4, 4)) +
                                     '), Time: ' + str(int((toc - tic) * 100)) + 's')


                # with torch.no_grad():
                test_loss_epoch=0
                u_batch_pred_test=None
                for batch_idx, (features, targets) in enumerate(data_loader_test):
                    if dist.is_initialized() and dist.get_world_size()>1:
                        u_batch_pred_test = dop.forward(features, dop.module.trunk_inputs)
                    else:
                        u_batch_pred_test = dop.forward(features, dop.trunk_inputs)

                    if(len(u_batch_pred_test.shape)==3):
                        u_batch_pred_test = u_batch_pred_test[0]


                    loss_batch = self.losses(targets[0], u_batch_pred_test)
                    test_loss_epoch += loss_batch #* batches[batch_index, 2] # TODO:: could be fixed


                if dist.is_initialized() and dist.get_world_size()>1:
                    with torch.no_grad():
                        dist.all_reduce(test_loss_epoch, op=dist.ReduceOp.SUM)
                        test_loss_epoch = 1./dist.get_world_size()*test_loss_epoch
                        test_loss_epoch = to_numpy(test_loss_epoch)
                else:
                    test_loss_epoch = to_numpy(test_loss_epoch)


                loss_test[epoch_index] = test_loss_epoch/(batch_idx+1)



                if ( dist.is_initialized()==False or (dist.is_initialized() and dist.get_rank()==0)):
                    if should_save and (loss_test[epoch_index]< best_test_loss):

                        if(dist.is_initialized()):
                            self.save_model(dop, args.model_name + "_"+str(dist.get_world_size()))
                        else:
                            self.save_model(dop, args.model_name)

                        logger.info("Saving model, new test loss: " + str( loss_test[epoch_index]))
                        best_test_loss = loss_test[epoch_index]

                        self.non_saving_count = 0

                    else:
                        self.non_saving_count  += 1


            # if ( dist.is_initialized()==False or (dist.is_initialized() and dist.get_rank()==0)):
            #     if plot_interval > 0 and (epoch_index + 1) % plot_interval == 0:

            #         with torch.no_grad():
            #             cleanup_folder('debug_figs/loss'+str(args.model_name)+'_'+str(args.dataset_name))

            #             fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(3.2 * 1, 2.4 * 1))

            #             if(loss_train.dtype == torch.complex64):
            #                 plt.semilogy(np.flatnonzero(loss_train.real), loss_train.real[loss_train.real!=0], axes=axs, label='Train loss')
            #                 plt.semilogy(plot_interval*np.flatnonzero(loss_test.real[loss_test.real<9e8]), loss_test.real[loss_test.real<9e8], axes=axs,  label='Test loss')                            
            #             else:
            #                 plt.semilogy(np.flatnonzero(loss_train), loss_train[loss_train!=0], axes=axs, label='Train loss')
            #                 plt.semilogy(plot_interval*np.flatnonzero(loss_test[loss_test<9e8]), loss_test[loss_test<9e8], axes=axs,  label='Test loss')
            #             # plt.semilogy(np.flatnonzero(loss_train), loss_train[loss_train!=0], axes=axs, label='Train loss')
            #             # plt.semilogy(np.flatnonzero(loss_test), loss_test[loss_test!=0], axes=axs,  label='Test loss')
                        
            #             plt.legend()
            #             plt.grid()

            #             plt.savefig("debug_figs/loss"+str(args.model_name)+'_'+str(args.dataset_name)+"/loss_"+str(epoch_index)+".png")
            #             plt.close()




        if (dist.is_initialized()==False or (dist.is_initialized() and dist.get_rank()==0)):
            if should_save and (loss_test[epoch_index]< best_test_loss and self.non_saving_count < args.max_stall/args.print_interval):
            #if should_save:
                if(dist.is_initialized()):
                    self.save_model(dop, args.model_name + "_"+str(dist.get_world_size()))
                else:
                    self.save_model(dop, args.model_name)

                logger.info("Saving model, new test loss: "+ str( loss_test[epoch_index]))


                best_test_loss = loss_test[epoch_index]
            else:
                logger.info("Exiting training without updating params of the model at the end... ")









