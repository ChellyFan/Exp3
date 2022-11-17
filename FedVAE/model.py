"""## Federated VAE"""
import os
import numpy as np
import pandas as pd
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
# from tqdm.notebook import tqdm
from tqdm import tqdm, trange

# from rectorch.models import MultiVAE
from rectorch.samplers import DataSampler
from rectorch.nets import VAE_net
from rectorch.evaluation import evaluate, ValidFunc


class MultiVAE_net(VAE_net):
    r'''Variational Autoencoder network for collaborative filtering.

    The network structure follows the definition as in [VAE]_. Hidden layers are fully
    connected and *tanh* activated. The output layer of both the encoder and the decoder
    are linearly activated.

    Parameters
    ----------
    dec_dims : :obj:`list` or array_like of :obj:`int`
        See :class:`AE_net`.
    enc_dims : :obj:`list`, array_like of :obj:`int` or None [optional]
        See :class:`AE_net`.
    dropout : :obj:`float`, optional
        See :class:`VAE_net`

    Attributes
    ----------
    dec_dims : :obj:`list` or array_like of :obj:`int`
        See :attr:`dec_dims` parameter.
    enc_dims : :obj:`list` or array_like of :obj:`int`
        See :attr:`end_dims` parameter.
    dropout : :obj:`float`
        The dropout layer that is applied to the input during the :meth:`VAE_net.forward`.

    References
    ----------
    .. [VAE] Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara. 2018.
       Variational Autoencoders for Collaborative Filtering. In Proceedings of the 2018
       World Wide Web Conference (WWW ’18). International World Wide Web Conferences Steering
       Committee, Republic and Canton of Geneva, CHE, 689–698.
       DOI: https://doi.org/10.1145/3178876.3186150
    '''

    def __init__(self, dec_dims, enc_dims=None, dropout=0.5):
        super(MultiVAE_net, self).__init__(dec_dims, enc_dims)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x, dp=None):
        # dp is fixed dropout
        h = F.normalize(x)
        # if self.training:
        h = self.dropout(h) if dp is None else h * dp
        for i, layer in enumerate(self.enc_layers):
            h = layer(h)
            if i != len(self.enc_layers) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.enc_dims[-1]]
                logvar = h[:, self.enc_dims[-1]:]
        return mu, logvar

    def _reparameterize(self, mu, logvar, eps=None):
        if self.training:
            # return super()._reparameterize(mu, logvar)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) if eps is None else eps
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        h = z
        for _, layer in enumerate(self.dec_layers[:-1]):
            h = torch.tanh(layer(h))
        return self.dec_layers[-1](h)
    
    def forward(self, x, dp=None, eps=None):
        r"""Apply the full Variational Autoencoder network to the input.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            The input tensor

        Returns
        -------
        x', mu, logvar : :obj:`tuple` of :py:class:`torch.Tensor`
            The reconstructed input (x') along with the intermediate tensors in the latent space
            representing the mean and standard deviation (actually the logarithm of the variance)
            of the probability distributions over the latent variables.
        """
        mu, logvar = self.encode(x, dp)
        z = self._reparameterize(mu, logvar, eps)
        return self.decode(z), mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return BCE + beta * KLD


class Federated_Sampler(DataSampler):
    def __init__(self,
                 sparse_data_tr,
                 sparse_data_te=None,
                 mode='train',
                 batch_size=64,
                 seed=98765):
        super(Federated_Sampler, self).__init__(sparse_data_tr, sparse_data_te, batch_size, False)
        self.mode = mode
        self.idxlist = list(range(self.sparse_data_tr.shape[0]))
        np.random.seed(seed)

    def __len__(self):
        return int(np.ceil(self.sparse_data_tr.shape[0] / self.batch_size))

    def set_mode(self, mode="train", batch_size=None):
        if mode == 'train':
            self.idxlist = list(range(self.sparse_data_tr.shape[0]))
        elif mode == 'valid':
            assert self.sparse_data_te is not None
            self.idxlist = list(range(self.sparse_data_te[0].shape[0]))
        self.mode = mode

    def __iter__(self):
        if self.mode == "train":
            n = self.sparse_data_tr.shape[0]
            for _, start_idx in enumerate(range(0, n, self.batch_size)):
                np.random.shuffle(self.idxlist)
                data_tr = self.sparse_data_tr[self.idxlist[:self.batch_size]]
                yield torch.FloatTensor(data_tr.toarray()), None
        else:
            n = self.sparse_data_te[0].shape[0]
            for _, start_idx in enumerate(range(0, n, self.batch_size)):
                end_idx = min(start_idx + self.batch_size, n)
                data_te_tr = torch.FloatTensor(self.sparse_data_te[0][self.idxlist[start_idx:end_idx]].toarray())
                data_te_te = torch.FloatTensor(self.sparse_data_te[1][self.idxlist[start_idx:end_idx]].toarray())

            yield data_te_tr, data_te_te


class FedMultiVAE:
    def __init__(self, dataset, dropout=0.5):
        torch.manual_seed(98765)
        self.network = MultiVAE_net(dec_dims=[200, 600, dataset.n_items],
                       enc_dims=None,
                       dropout=dropout).to(torch.device("cuda"))
        self.device = torch.device("cuda")
        self.optimizer = Adam(self.network.parameters(), lr=0.001)
        # gradient record
        self.df_ng = pd.DataFrame(columns=['v0', 'v1', 'e'])
        self.df_s = pd.DataFrame(columns=['v0', 'v1', 'e'])
        self.df_g = pd.DataFrame(columns=['p_e', 'p_f', 'g'])

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return BCE + beta * KLD

    def gradient_step(self, network, data_tensor, dp=None, eps=None):
        network.train()
        recon_batch, mu, var = network(data_tensor, dp, eps)
        loss = self.loss_function(recon_batch, data_tensor, mu, var, beta=.2)
        loss.backward()

    def sum_grad(self, net1, net2):
        for pA, pB in zip(net1.parameters(), net2.parameters()):
            pB.grad = pA.grad + (pB.grad if pB.grad is not None else 0)

    def train_epoch(self, epoch, data_sampler: Federated_Sampler, record_grad_info):
        data_sampler.set_mode("train")
        # for data, _ in tqdm(data_sampler):
        for i, (data, _) in enumerate(tqdm(data_sampler, desc="Epoch %d" %(epoch))):
            self.optimizer.zero_grad()
            data_tensor = data.view(data.shape[0], -1).to(self.device)
            if record_grad_info and i == 0:
                idx = torch.sum(data_tensor, dim=1).argmax().item()
            
            for u in range(len(data)):
                net_clone = copy.deepcopy(self.network)
                self.gradient_step(net_clone, data_tensor[u:u+1])
                if record_grad_info and i == 0 and u == idx:
                    self.record_norm_grad(epoch, net_clone, data_tensor[u:u+1][0])
                    self.record_sim(epoch, net_clone, data_tensor[u:u+1][0])
                    self.record_grad(epoch, net_clone, data_tensor[u:u+1][0])
                self.sum_grad(net_clone, self.network)
            
            for p in self.network.parameters():
                p.grad /= len(data_sampler.idxlist)#len(data)

            self.optimizer.step()

            # recon_batch, mu, var = self.network(data_tensor)
            # loss = self.loss_function(recon_batch, data_tensor, mu, var, beta=.2)
            # print("Loss: ", loss.item())
        data_sampler.set_mode("valid")
        return ValidFunc(evaluate)(self, data_sampler, "ndcg@100")

    def train(self, data_sampler, num_epochs=100, record_grad_info=False):
        try:
            for e in range(num_epochs):
                # print("Epoch %d" %(e+1))
                valid_res = self.train_epoch(e+1, data_sampler, record_grad_info=(record_grad_info and (True if ((e + 1) % 20) == 0 else False)))
                print("ndcg@100:", np.mean(valid_res))
        except KeyboardInterrupt:
            print("Interrupted!")

    def predict(self, x, remove_train=True):
        self.network.eval()
        with torch.no_grad():
            x_tensor = x.to(self.device)
            recon_x, mu, logvar = self.network(x_tensor)
            if remove_train:
                recon_x[torch.nonzero(x_tensor, as_tuple=True)] = -np.inf
            return recon_x, mu, logvar
        
    def record_norm_grad(self, epoch, net, data_u):
        g_norm = torch.norm(list(net.parameters())[-2].grad, p=2, dim=1).cpu().numpy()
        select = data_u.cpu().numpy().astype(bool)
        v0 = copy.deepcopy(g_norm)
        v0[select] = np.nan
        v1 = copy.deepcopy(g_norm)
        v1[np.logical_not(select)] = np.nan
        epoch_list = [epoch] * len(g_norm)
        df_n = pd.DataFrame(dict(v0=v0, v1=v1, e=epoch_list))
        self.df_ng = self.df_ng.append(df_n)
        print("record norm of gradient.")
        
    def record_sim(self, epoch, net, data_u):
        g = list(net.parameters())[-2].grad
        idx = data_u.argmax().item()    # idx of a positive label
        sim = torch.sum(g[idx] * g, dim=1) / (torch.norm(g[idx], p=2) * torch.norm(g, p=2, dim=1))
        sim = sim.cpu().numpy()
        select = data_u.cpu().numpy().astype(bool)
        v0 = copy.deepcopy(sim)
        v0[select] = np.nan
        v1 = copy.deepcopy(sim)
        v1[np.logical_not(select)] = np.nan
        epoch_list = [epoch] * len(sim)
        df_n = pd.DataFrame(dict(v0=v0, v1=v1, e=epoch_list))
        self.df_s = self.df_s.append(df_n)
        print("record similarity of gradient.")
        
    def record_grad(self, epoch, net, data_u):
        pos_enc = torch.unique(list(net.parameters())[0].grad.nonzero(as_tuple=True)[1]).cpu().numpy().tolist()
        pos_full = data_u.nonzero(as_tuple=True)[0].cpu().numpy().tolist()
        g = list(net.parameters())[-2].grad.cpu().numpy().tolist()
        # df_n = pd.DataFrame(dict(p_e=pos_enc, p_f=pos_full, g=g))
        self.df_g = self.df_g.append(dict(p_e=pos_enc, p_f=pos_full, g=g), ignore_index=True)
        print("record gradient and positive items.")
        
    def save_record_norm_grad(self, save_path):
        self.df_ng.to_csv(os.path.join(save_path, 'norm_grad.csv'), sep='\t', index=False)
        
    def save_record_similarity(self, save_path):
        self.df_s.to_csv(os.path.join(save_path, 'similarity.csv'), sep='\t', index=False)
        
    def save_record_grad(self, save_path):
        self.df_g.to_csv(os.path.join(save_path, 'gradient.csv'), sep='\t', index=False)
        
        
class FedMomMultiVAE(FedMultiVAE):
    def __init__(self, mom_beta=0.9, discount=1):
        super(FedMomMultiVAE, self).__init__()
        self.mom_beta = mom_beta
        self.discount = discount
    
    def exp_discount(self, bmin, bstart, x):
        return bmin + (bstart - bmin)*(1 - self.discount)**x
    
    def momentum(self, w, v, beta=0.9):
        with torch.no_grad():
            for lw, lv in zip(w.enc_layers, v.enc_layers):
                lw.weight += beta * (lw.weight - lv.weight)

            for lw, lv in zip(w.dec_layers, v.dec_layers):
                lw.weight += beta * (lw.weight - lv.weight)

    def train_epoch(self, epoch, data_sampler, valid_sampler):
        for data, _ in tqdm(data_sampler):
            self.optimizer.zero_grad()
            if self.mom_beta > 0:
                v_t = copy.deepcopy(self.network)
            data_tensor = data.view(data.shape[0], -1).to(self.device)
            for u in range(len(data)):
                net_clone = copy.deepcopy(self.network)
                self.gradient_step(net_clone, data_tensor[u:u+1])
                self.sum_grad(net_clone, self.network)
            
            for p in self.network.parameters():
                p.grad /= len(data_sampler.idxlist)
            
            self.optimizer.step()

            if self.mom_beta > 0:
                v_t_prev = copy.deepcopy(self.network)
                mom = self.exp_discount(0., self.mom_beta, epoch-1)
                self.momentum(self.network, v_t, mom)
                v_t = v_t_prev

            recon_batch, mu, var = self.network(data_tensor)
            loss = self.loss_function(recon_batch, data_tensor, mu, var, beta=0.2)
            print("Loss: ", loss.item())
        
        return ValidFunc(evaluate)(self, valid_sampler, "ndcg@100")