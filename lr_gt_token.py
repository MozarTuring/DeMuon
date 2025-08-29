import os
import random
import pickle
from dataclasses import dataclass, field

import torch 
from torch.distributions import StudentT
import torch.nn.init as init
from tqdm import tqdm

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.stats import levy_stable
# from torchlevy import LevyStable, stable_dist # a user-defined package for Levy stable distribution 

from toolbox import smooth_clipping, dump_pickle
from optim_network import * 

from token_data import token_data_generate
from regression_model import LinearRegressionModel, TukeyBiweightLoss


class LrNoncvx(object):
    '''
    This class defines the problem and the gradient oracle for a distributed non-convex linear regression problem. 
    '''
    def __init__(self, data, num_clients, seed=24):
        self.data = data # data[0]; X, data[1] y. 
        self.num_clients = num_clients
        self.data_distributed = self.distribute_data()
        torch.manual_seed(seed)
        self.model_distributed = self.distribute_model()
        self.criterion = TukeyBiweightLoss()
        # this loss has flat region, 

    ### data defined problem info and sg oracles
    def distribute_data(self):
        '''
        distributing datasets to all clients evenly (almost)
        '''
        N = len(self.data[0]) 
        n = round(N / self.num_clients)
        data_distributed = [] 
        for idx in range(self.num_clients):
            start = idx * n
            if idx != self.num_clients - 1:
                end = start + n 
            else:
                end = N
            data_distributed.append((self.data[0][start:end], self.data[1][start:end]))
        return data_distributed
    
    def distribute_model(self):
        model_distributed = []
        for idx in range(self.num_clients):
            model = LinearRegressionModel(input_dim=self.data[0].shape[1])
            init.zeros_(model.linear.weight)
            model_distributed.append(model)
        # initialize all models with the same parameters
        init_state = model_distributed[0].state_dict()
        for idx in range(1, self.num_clients):
            model_distributed[idx].load_state_dict(init_state)
        return model_distributed

    def distributed_gd(self):
        """
        For each (model, (X, y)) pair:
        - Zero existing grads
        - Forward pass => compute loss
        - Backward => fill param.grad
        - Collect these grads in a dict
        Return: a list of gradient dicts, one for each model.

        model_list   : list of PyTorch models (e.g. MyLinearRegression instances)
        dataset_list : list of tuples (X_i, y_i), each a local dataset for model i
        criterion    : a loss function, e.g. nn.MSELoss or custom

        Requirements:
        - X_i, y_i are torch.Tensors, shapes match the model's forward pass
        - len(model_list) == len(dataset_list)
        """
        all_grad_dicts = []

        for i, model in enumerate(self.model_distributed):
            # 1) Zero-out old grads
            model.zero_grad()

            # 2) Unpack local data
            X_i, y_i = self.data_distributed[i]
            # Ensure X_i, y_i are on correct device if using CUDA, etc.

            # 3) Forward pass
            pred = model(X_i)            # shape (batch_size,1) or (batch_size,)
            loss = self.criterion(pred, y_i)  # scalar

            # 4) Backward pass => populates param.grad
            loss.backward()

            # 5) Copy out the gradients into a dict
            grad_dict = {}
            for name, param in model.named_parameters():
                # param.grad is None if param is frozen (requires_grad=False)
                if param.grad is not None:
                    grad_dict[name] = param.grad.detach().clone()
                else:
                    grad_dict[name] = None

            all_grad_dicts.append(grad_dict)
        all_grads = [grad_dict['linear.weight'] for grad_dict in all_grad_dicts]
        return torch.cat(all_grads, dim=0)
    
    def distributed_loss(self):
        '''
        Compute the loss for each model and return a list of losses.
        '''
        losses = []
        for i, model in enumerate(self.model_distributed):  
            X_i, y_i = self.data_distributed[i]
            model.eval()
            pred = model(X_i)
            loss = self.criterion(pred, y_i)
            losses.append(loss)
        return losses

    def distributed_optim_gap(self, w_true):
        '''
        Compute the optimality gap for each model and return a list of optimality gaps.
        '''
        gaps = []
        for i, model in enumerate(self.model_distributed):
            gaps.append(la.norm(model.state_dict()['linear.weight'] - w_true))
        return gaps
    
    def global_grad(self):
        all_grad_dicts = []
        X, y = self.data
        for i, model in enumerate(self.model_distributed):
            # 1) Zero-out old grads
            model.zero_grad()

            # 3) Forward pass
            pred = model(X)            # shape (batch_size,1) or (batch_size,)
            loss = self.criterion(pred, y)  # scalar

            # 4) Backward pass => populates param.grad
            loss.backward()

            # 5) Copy out the gradients into a dict
            grad_dict = {}
            for name, param in model.named_parameters():
                # param.grad is None if param is frozen (requires_grad=False)
                if param.grad is not None:
                    grad_dict[name] = param.grad.detach().clone()
                else:
                    grad_dict[name] = None

            all_grad_dicts.append(grad_dict)
        all_grads = [grad_dict['linear.weight'] for grad_dict in all_grad_dicts]
        return torch.cat(all_grads, dim=0)
    
    def global_loss(self):
        X, y = self.data
        pred = self.model_distributed[0](X)
        loss = self.criterion(pred, y)
        return loss


def run_datasets_alg_network(dataset: str, noise: str, noise_scale: float, alg: str, paras,  
                             num_rep: int, num_steps: int, seed: int=24, num_clients: int=20) -> None:
    '''
    given dataset, run alg for num_epochs, each epoch is divided into num_batch mini-batches,
    and this is repeated num_rep times. dataset is str which is used to fetch data. alg is 
    '''
    # dataset
    if not os.path.exists('../datasets'):
        os.mkdir('../datasets')
    if not os.path.exists('../datasets/syntoken_data.npz'):
        token_data_generate()
    data = np.load('../datasets/syntoken_data.npz', allow_pickle=True)
    w_true = torch.from_numpy(data['w_true']).float() # numpy uses float64, torch uses float32
    X = torch.from_numpy(data[f'X_{dataset.split("_")[0]}']).float()
    y = torch.from_numpy(data[f'y_{dataset}']).float()
    
    # train num_epochs epochs, and repat num_rep times
    eval_interval = 100
    # num_clients = 20 # this suits with the num_samples
    stats = np.zeros((3, num_steps // eval_interval + 1)) # mean, min, max 
    # breakpoint()
    graph_weights = torch.from_numpy(paras[0]).float()
    if noise == 'gaussian':
        noise_dist = torch.distributions.Normal(0, noise_scale)
    elif noise == 'student':
        noise_dist = torch.distributions.StudentT(df=noise_scale, loc=0.0, scale=1.0)
    for rep in range(num_rep):
        # initialize the model for each repetition independently
        runners = LrNoncvx((X, y), num_clients, seed=seed+rep)
        w0 = runners.model_distributed[0].state_dict()['linear.weight']
        X = w0.repeat(num_clients, 1)
        error0 = torch.linalg.norm(w0-w_true) # use average gd norm for nonconvex functions
        x_errs = [error0]
        step = 0 
        for step in range(num_steps): 
            SG_X = runners.distributed_gd()
            if noise == 'gaussian' or noise == 'student':
                SG_X.add_(noise_dist.sample(SG_X.shape))
            elif noise == 'levy':
                levy_noise = stable_dist.sample(alpha=1.5, beta=0.5, size=SG_X.shape, loc=0, scale=noise_scale) 
                levy_noise[torch.isnan(levy_noise)] = 0.0 # remove NaNs
                # levy_noise.clamp_(-1, 1).mul_(noise_scale) # clip to [-1, 1] and scale to 0.1 times the original scale
                # levy_noise = levy_noise.mul_(noise_scale)
                SG_X.add_(levy_noise)
            if alg == 'dsgd':
                alpha = paras[1]
                X = dsgd(graph_weights, X, alpha, SG_X)
            elif alg == 'dsgd_gclip_decay': # decaying rate chosen from https://arxiv.org/pdf/2312.15847
                alpha, l2_bd = paras[1:]
                alpha = alpha/(step+1)
                l2_bd = l2_bd/(step+1)**0.4
                X = dsgd_gclip(graph_weights, X, alpha, SG_X, l2_bd)
            elif alg == 'gt_dsgd':
                x_stepsize = paras[1]
                if step == 0: 
                    Y = torch.zeros((num_clients, len(w_true)))
                    SG_pre = torch.zeros((num_clients, len(w_true)))
                X, Y, SG_pre = gt_dsgd(graph_weights, X, Y, SG_pre, x_stepsize, SG_X)
            elif alg == 'gt_nsgdm':
                x_stepsize, m_stepsize = paras[1:]
                if step == 0: 
                    Y = torch.zeros((num_clients, len(w_true)))
                    M = torch.zeros((num_clients, len(w_true)))
                X, Y, M = gt_nsgdm(graph_weights, X, Y, M, x_stepsize, m_stepsize, SG_X)
            elif alg == 'sclip_ef_network':
                x_stepsize_const, m_stepsize_const, phi_const, eps_const = paras[1:]
                x_stepsize = x_stepsize_const/(step+1)**0.2
                m_stepsize = m_stepsize_const/(step+1)**0.5
                phi = phi_const/(step+1)**0.5
                eps = eps_const*(step+1)**0.6
                if step == 0: M = torch.zeros((num_clients, len(w_true)))
                X, M = sclip_ef_network(graph_weights, X, M, x_stepsize, m_stepsize, phi, eps, SG_X)
            else:
                raise Exception('Algorithm not defined')

            # apply updated weights to model
            for i, model in enumerate(runners.model_distributed):
                with torch.no_grad():
                    model.linear.weight.copy_(X[i])
            if (step + 1) % eval_interval == 0:
                x_errs.append(np.mean(runners.distributed_optim_gap(w_true)))
        # get stats for all repetitions
        # stats[0] = (rep*stats[0] + x_errs)/(rep+1)
        # stats[1] = x_errs if rep == 0 else np.minimum(stats[1], x_errs)
        # stats[2] = x_errs if rep == 0 else np.maximum(stats[2], x_errs)
        if rep == 0:
            stats[0] = x_errs.copy()          # running mean starts here
            stats[1] = x_errs.copy()          # min tracker
            stats[2] = x_errs.copy()          # max tracker
        else:
            stats[0] = (rep * stats[0] + x_errs) / (rep + 1)
            stats[1] = np.minimum(stats[1], x_errs)
            stats[2] = np.maximum(stats[2], x_errs)
        plt.plot(np.arange(len(x_errs))*eval_interval, x_errs, markevery=10, lw=0.5,linestyle='-', label=f'alpha{paras[1]}_beta{paras[2]}_{alg}')
    return stats
