import sys
import torch
import numbers
from torch.nn.parameter import Parameter
from time import sleep
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.nn.functional import normalize
from torch.nn.utils import spectral_norm
from torch_utils import *

from tensorboardX import SummaryWriter
import os
import numpy as np
import math

n_eigens_debug = 20
# save_dir = 'runs'
# logdir = os.path.join(save_dir, 'ResNet18_adv_adapt', '{}-bs{}'.format('pcanorm', 128),
#                       'pi-eig{}-trail2'.format(n_eigens_debug))
# writer = SummaryWriter(log_dir=logdir)
# global step
# step = 0


# def save_grad_(grad):
#     global step
#     writer.add_scalar('grad/{}mean'.format('dL_dM_'), grad.abs().mean().item(), step)
#     writer.add_scalar('grad/{}max'.format('dL_dM_'), grad.abs().max().item(), step)
#     step += 1

def print_grad(grad):
    print('hi')
    print(grad)
    sleep(5)


class ZCANormPIv2Clip(nn.Module):
    def __init__(self, num_features, groups=16, eps=1e-2, momentum=0.1, affine=True):
        super(ZCANormPIv2Clip, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.groups = groups
        self.weight = Parameter(torch.Tensor(num_features, 1))
        self.bias = Parameter(torch.Tensor(num_features, 1))

        self.svdlayer = svdv2clip.apply

        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = int(self.num_features / self.groups)
        for i in range(self.groups):
            self.register_buffer("running_subspace{}".format(i), torch.eye(length, length))

    def reset_running_stats(self):
            self.running_mean.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        if self.training:
            N, C, H, W = x.size()
            G = self.groups
            x = x.transpose(0, 1).contiguous().view(C, -1)
            mu = x.mean(1, keepdim=True)
            x = x - mu
            xxt = torch.mm(x, x.t()) / (N * H * W) + torch.eye(C, out=torch.empty_like(x)) * self.eps
            assert C % G == 0
            length = int(C/G)
            xxti = torch.chunk(xxt, G, dim=0)
            xxtj = [torch.chunk(xxti[j], G, dim=1)[j] for j in range(G)]

            xg = list(torch.chunk(x, G, dim=0))
            xgr_list = []
            for i in range(G):
                v, e = self.svdlayer(xxtj[i])
                # ratio = torch.cumsum(e, 0)/e.sum()
                # rank = 0
                # for j in range(length):
                #     if e[j] <= self.eps or ratio[j] >= (1 - self.eps):
                #         print('{}/{} eigen-vectors selected'.format(rank+1, length))
                #         # print(e[0:rank])
                #         break
                #     # lambda_ij = torch.mm(xxtj[i].mm(v[:, j][..., None]).t(), v[:, j][..., None])/torch.mm(v[:, j][..., None].t(), v[:, j][..., None])
                #     # if lambda_ij < 0:
                #     #     print('eigenvalues: ', e)
                #     #     print("Warning message: negative lambda_ij {} vs SVD lambda_ij {}..".format(lambda_ij, e[j]))
                #     #     break
                #     # diff_ratio = (lambda_ij - e[j]).abs()/e[j]
                #     # if diff_ratio > 0.1:
                #     #     print('inaccurate eigenvalue computed: ', e)
                #     #     print("Warning message: inaccurate lambda_ij {} vs SVD lambda_ij {}..".format(lambda_ij, e[j]))
                #     #     break
                #     rank += 1
                # rank = int(np.clip(rank, 1, length))
                # ############### truncate eigenvectors ##################
                # rank = 16
                # v = v[:, 0:rank]
                # e = e[0:rank]
                # ########################################################
                # e = torch.diag(torch.mm(v.t(), xxtj[i].mm(v)))
                subspace = torch.mm(v, torch.diag(torch.rsqrt(e)).mm(v.t()))
                xgr = torch.mm(subspace, xg[i])
                xgr_list.append(xgr)

                with torch.no_grad():
                    running_subspace = self.__getattr__('running_subspace' + str(i))
                    running_subspace.data = (1 - self.momentum) * running_subspace.data + self.momentum * subspace.data

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu

            xr = torch.cat(xgr_list, dim=0)
            xr = xr * self.weight + self.bias
            xr = xr.view(C, N, H, W).transpose(0, 1)

            return xr

        else:
            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            x = (x - self.running_mean)
            G = self.groups
            xg = list(torch.chunk(x, G, dim=0))
            for i in range(G):
                subspace = self.__getattr__('running_subspace' + str(i))
                xg[i] = torch.mm(subspace, xg[i])
            x = torch.cat(xg, dim=0)
            x = x * self.weight + self.bias
            x = x.view(C, N, H, W).transpose(0, 1)
            return x

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(ZCANormPIv2Clip, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

class ZCANormPIv2(nn.Module):
    def __init__(self, num_features, groups=1, eps=1e-2, momentum=0.1, affine=True):
        super(ZCANormPIv2, self).__init__()
        print('ZCANormPIv2 Group Num {}'.format(groups))
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.groups = groups
        self.weight = Parameter(torch.Tensor(num_features, 1))
        self.bias = Parameter(torch.Tensor(num_features, 1))

        self.svdlayer = svdv2.apply

        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = int(self.num_features / self.groups)
        for i in range(self.groups):
            self.register_buffer("running_subspace{}".format(i), torch.eye(length, length))

    def reset_running_stats(self):
            self.running_mean.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        if self.training:
            N, C, H, W = x.size()
            G = self.groups
            x = x.transpose(0, 1).contiguous().view(C, -1)
            mu = x.mean(1, keepdim=True)
            x = x - mu
            xxt = torch.mm(x, x.t()) / (N * H * W) + torch.eye(C, out=torch.empty_like(x)) * self.eps
            assert C % G == 0
            length = int(C/G)
            xxti = torch.chunk(xxt, G, dim=0)
            xxtj = [torch.chunk(xxti[j], G, dim=1)[j] for j in range(G)]

            xg = list(torch.chunk(x, G, dim=0))
            xgr_list = []
            for i in range(G):
                v, e = self.svdlayer(xxtj[i])
                # ratio = torch.cumsum(e, 0)/e.sum()
                # rank = 0
                # for j in range(length):
                #     if e[j] <= self.eps or ratio[j] >= (1 - self.eps):
                #         print('{}/{} eigen-vectors selected'.format(rank+1, length))
                #         # print(e[0:rank])
                #         break
                #     # lambda_ij = torch.mm(xxtj[i].mm(v[:, j][..., None]).t(), v[:, j][..., None])/torch.mm(v[:, j][..., None].t(), v[:, j][..., None])
                #     # if lambda_ij < 0:
                #     #     print('eigenvalues: ', e)
                #     #     print("Warning message: negative lambda_ij {} vs SVD lambda_ij {}..".format(lambda_ij, e[j]))
                #     #     break
                #     # diff_ratio = (lambda_ij - e[j]).abs()/e[j]
                #     # if diff_ratio > 0.1:
                #     #     print('inaccurate eigenvalue computed: ', e)
                #     #     print("Warning message: inaccurate lambda_ij {} vs SVD lambda_ij {}..".format(lambda_ij, e[j]))
                #     #     break
                #     rank += 1
                # rank = int(np.clip(rank, 1, length))
                # ############### truncate eigenvectors ##################
                # rank = 16
                # v = v[:, 0:rank]
                # e = e[0:rank]
                # ########################################################
                # e = torch.diag(torch.mm(v.t(), xxtj[i].mm(v)))
                subspace = torch.mm(v, torch.diag(torch.rsqrt(e)).mm(v.t()))
                xgr = torch.mm(subspace, xg[i])
                xgr_list.append(xgr)

                with torch.no_grad():
                    running_subspace = self.__getattr__('running_subspace' + str(i))
                    running_subspace.data = (1 - self.momentum) * running_subspace.data + self.momentum * subspace.data

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu

            xr = torch.cat(xgr_list, dim=0)
            xr = xr * self.weight + self.bias
            xr = xr.view(C, N, H, W).transpose(0, 1)

            return xr

        else:
            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            x = (x - self.running_mean)
            G = self.groups
            xg = list(torch.chunk(x, G, dim=0))
            for i in range(G):
                subspace = self.__getattr__('running_subspace' + str(i))
                xg[i] = torch.mm(subspace, xg[i])
            x = torch.cat(xg, dim=0)
            x = x * self.weight + self.bias
            x = x.view(C, N, H, W).transpose(0, 1)
            return x

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(ZCANormPIv2, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class ZCANormOrg(nn.Module):
    def __init__(self, num_features, groups=4, eps=1e-4, momentum=0.1, affine=True):
        super(ZCANormOrg, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.groups = groups

        self.weight = Parameter(torch.Tensor(num_features, 1))
        self.bias = Parameter(torch.Tensor(num_features, 1))
        self.power_layer = power_iteration_once.apply
        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = int(self.num_features / self.groups)
        for i in range(self.groups):
            self.register_buffer("running_subspace{}".format(i), torch.eye(length, length))

    def reset_running_stats(self):
            self.running_mean.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        if self.training:
            N, C, H, W = x.size()
            G = self.groups
            x = x.transpose(0, 1).contiguous().view(C, -1)
            mu = x.mean(1, keepdim=True)
            x = x - mu
            xxt = torch.mm(x, x.t())/(N*H*W) + torch.eye(C, out=torch.empty_like(x)) * self.eps

            assert C % G == 0
            length = int(C / G)
            xxti = torch.chunk(xxt, G, dim=0)
            xxtj = [torch.chunk(xxti[j], G, dim=1)[j] for j in range(G)]

            xg = list(torch.chunk(x, G, dim=0))

            xgr_list = []
            for i in range(G):
                u, e, v = torch.svd(xxtj[i])
                ratio = torch.cumsum(e, 0) / e.sum()
                counter_i = 1
                for j in range(length):
                    if e[j] <= self.eps:  # ratio[j] >= (1 - self.eps) or e[j] <= self.eps:
                        # print('{}/{} eigen-vectors selected'.format(j + 1, length))
                        counter_i = j + 1  # at least keep 99.99% energy
                        break
                subspace = torch.zeros_like(xxtj[i])
                for j in range(counter_i):
                    lambda_ij = e[j]
                    if lambda_ij < 0:
                        print('eigenvalues: ', e)
                        print("Error message: negative SVD lambda_ij {} vs SVD lambda_ij {}..".format(lambda_ij, e[j]))
                        break
                    eigenvector_ij = v[:, j][..., None]
                    subspace += torch.mm(eigenvector_ij, torch.rsqrt(lambda_ij) * eigenvector_ij.t())
                xgr = torch.mm(subspace, xg[i])
                xgr_list.append(xgr)

                with torch.no_grad():
                    running_subspace = self.__getattr__('running_subspace' + str(i))
                    running_subspace.data = (1 - self.momentum) * running_subspace.data + self.momentum * subspace.data
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu

            xr = torch.cat(xgr_list, dim=0)
            xr = xr * self.weight + self.bias
            xr = xr.view(C, N, H, W).transpose(0, 1)
            return xr

                # v2 = torch.diag(torch.rsqrt(e)).mm(v.t())
                # proj = torch.mm(v, v2)
                # xg[i] = torch.mm(proj, xg[i])
                #
                # with torch.no_grad():
                #     subspace = self.__getattr__('running_subspace'+str(i))
                #     subspace.data = (1 - self.momentum) * subspace.data + self.momentum * proj.data

            # with torch.no_grad():
            #     self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            #
            # xr = torch.cat(xg, dim=0)
            # xr = xr * self.weight + self.bias
            # xr = xr.view(C, N, H, W).transpose(0, 1)
            # return xr

        else:
            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            x = (x - self.running_mean)
            G = self.groups
            xg = list(torch.chunk(x, G, dim=0))
            for i in range(G):
                subspace = self.__getattr__('running_subspace' + str(i))
                xg[i] = torch.mm(subspace, xg[i])
            x = torch.cat(xg, dim=0)
            x = x * self.weight + self.bias
            x = x.view(C, N, H, W).transpose(0, 1)
            return x

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(ZCANormOrg, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class ZCANormPI(nn.Module):
    def __init__(self, num_features, groups=4, eps=1e-10, momentum=0.1, affine=True):
        print('using zca with power iteration group num {}'.format(groups))
        super(ZCANormPI, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.groups = groups
        self.weight = Parameter(torch.Tensor(num_features, 1))
        self.bias = Parameter(torch.Tensor(num_features, 1))
        self.power_layer = power_iteration_once.apply
        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = int(self.num_features / self.groups)
        for i in range(self.groups):
            self.register_buffer("running_subspace{}".format(i), torch.eye(length, length))
            for j in range(length):
                self.register_buffer('eigenvector{}-{}'.format(i, j), torch.ones(length, 1))

    def reset_running_stats(self):
            self.running_mean.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        if self.training:
            N, C, H, W = x.size()
            G = self.groups
            x = x.transpose(0, 1).contiguous().view(C, -1)
            mu = x.mean(1, keepdim=True)
            x = x - mu
            xxt = torch.mm(x, x.t()) / (N * H * W) + torch.eye(C, out=torch.empty_like(x)) * self.eps

            assert C % G == 0
            length = int(C/G)
            xxti = torch.chunk(xxt, G, dim=0)
            xxtj = [torch.chunk(xxti[j], G, dim=1)[j] for j in range(G)]

            xg = list(torch.chunk(x, G, dim=0))

            xgr_list = []
            for i in range(G):
                counter_i = 0
                # print(xxtj[i])
                # compute eigenvectors of subgroups no grad
                with torch.no_grad():
                    u, e, v = torch.svd(xxtj[i])
                    ratio = torch.cumsum(e, 0)/e.sum()
                    for j in range(length):
                        if ratio[j] >= (1 - self.eps) or e[j] <= self.eps:
                            print('{}/{} eigen-vectors selected'.format(j + 1, length))
                            # print(e[0:counter_i])
                            break
                        eigenvector_ij = self.__getattr__('eigenvector{}-{}'.format(i, j))
                        eigenvector_ij.data = v[:, j][..., None].data
                        counter_i = j + 1

                # feed eigenvectors to Power Iteration Layer with grad and compute whitened tensor
                subspace = torch.zeros_like(xxtj[i])
                for j in range(counter_i):
                    eigenvector_ij = self.__getattr__('eigenvector{}-{}'.format(i, j))
                    eigenvector_ij = self.power_layer(xxtj[i], eigenvector_ij)
                    # eigenvector_ij.register_hook(print)
                    lambda_ij = torch.mm(xxtj[i].mm(eigenvector_ij).t(), eigenvector_ij)/torch.mm(eigenvector_ij.t(), eigenvector_ij)
                    if lambda_ij < 0:
                        print('eigenvalues: ', e)
                        # sys.exit("Error message: negative PI lambda_ij {} vs SVD lambda_ij {}..".format(lambda_ij, e[j]))
                        print("Warning message: negative PI lambda_ij {} vs SVD lambda_ij {}..".format(lambda_ij, e[j]))
                        break
                    diff_ratio = (lambda_ij - e[j]).abs()/e[j]
                    if diff_ratio > 0.1:
                        # print('inaccurate eigenvalue computed: ', e)
                        # sys.exit("Error message: inaccurate PI lambda_ij {} vs SVD lambda_ij {}..".format(lambda_ij, e[j]))
                        # print("Warning message: inaccurate PI lambda_ij {} vs SVD lambda_ij {}..".format(lambda_ij, e[j]))
                        break
                    subspace += torch.mm(eigenvector_ij, torch.rsqrt(lambda_ij).mm(eigenvector_ij.t()))
                    # print('lambda_group{}_{}: {} vs gt {}'.format(i, j, lambda_ij.item(), e[j].item()))
                    # if j>0:
                    #     print('SVD eig-{} (angle) between current and dominant eigenvectors'.format(j),
                    #           F.cosine_similarity(v[:, j][..., None], v[:, 0][..., None], dim=0).clamp(-1.0,1.0).abs().acos().item() / math.pi * 180)
                    # if lambda_ij<0:
                    #     import IPython
                    #     IPython.embed()
                    # remove projections on the eigenvectors
                    xxtj[i] = xxtj[i] - torch.mm(xxtj[i], eigenvector_ij.mm(eigenvector_ij.t()))
                xgr = torch.mm(subspace, xg[i])
                xgr_list.append(xgr)

                with torch.no_grad():
                    running_subspace = self.__getattr__('running_subspace' + str(i))
                    running_subspace.data = (1 - self.momentum) * running_subspace.data + self.momentum * subspace.data

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu

            xr = torch.cat(xgr_list, dim=0)
            xr = xr * self.weight + self.bias
            xr = xr.view(C, N, H, W).transpose(0, 1)

            return xr

        else:
            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            x = (x - self.running_mean)
            G = self.groups
            xg = list(torch.chunk(x, G, dim=0))
            for i in range(G):
                subspace = self.__getattr__('running_subspace' + str(i))
                xg[i] = torch.mm(subspace, xg[i])
            x = torch.cat(xg, dim=0)
            x = x * self.weight + self.bias
            x = x.view(C, N, H, W).transpose(0, 1)
            return x

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(ZCANormPI, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class ZCANormPI_fails_group_is_4(nn.Module):
    def __init__(self, num_features, groups=4, eps=1e-4, momentum=0.1, affine=True):
        super(ZCANormPI, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.groups = groups
        self.weight = Parameter(torch.Tensor(num_features, 1))
        self.bias = Parameter(torch.Tensor(num_features, 1))
        self.power_layer = power_iteration_once.apply
        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = int(self.num_features / self.groups)
        for i in range(self.groups):
            self.register_buffer("running_subspace{}".format(i), torch.eye(length, length))
            for j in range(length):
                self.register_buffer('eigenvector{}-{}'.format(i, j), torch.ones(length, 1))

    def reset_running_stats(self):
            self.running_mean.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        if self.training:
            N, C, H, W = x.size()
            G = self.groups
            x = x.transpose(0, 1).contiguous().view(C, -1)
            mu = x.mean(1, keepdim=True)
            x = x - mu
            xxt = torch.mm(x, x.t()) / (N * H * W) + torch.eye(C, out=torch.empty_like(x)) * self.eps

            assert C % G == 0
            length = int(C/G)
            xxti = torch.chunk(xxt, G, dim=0)
            xxtj = [torch.chunk(xxti[j], G, dim=1)[j] for j in range(G)]

            xg = list(torch.chunk(x, G, dim=0))

            xgr_list = []
            for i in range(G):
                counter_i = 0

                # compute eigenvectors of subgroups no grad
                with torch.no_grad():
                    u, e, v = torch.svd(xxtj[i])
                    ratio = torch.cumsum(e, 0)/e.sum()
                    for j in range(length):
                        counter_i = j+1
                        eigenvector_ij = self.__getattr__('eigenvector{}-{}'.format(i, j))
                        eigenvector_ij.data = v[:, j][..., None].data
                        if ratio[j] >= 1:
                            # print('{}/{} eigen-vectors selected'.format(j + 1, length))
                            break

                # feed eigenvectors to Power Iteration Layer with grad and compute whitened tensor
                subspace = torch.zeros_like(xxtj[i])
                for j in range(counter_i):
                    eigenvector_ij = self.__getattr__('eigenvector{}-{}'.format(i, j))
                    eigenvector_ij = self.power_layer(xxtj[i], eigenvector_ij)
                    # eigenvector_ij.register_hook(print)
                    lambda_ij = torch.mm(xxtj[i].mm(eigenvector_ij).t(), eigenvector_ij)/torch.mm(eigenvector_ij.t(), eigenvector_ij)
                    subspace += torch.mm(eigenvector_ij, torch.rsqrt(lambda_ij).mm(eigenvector_ij.t()))
                    # remove projections on the eigenvectors
                    xxtj[i] = xxtj[i] - torch.mm(xxtj[i], eigenvector_ij.mm(eigenvector_ij.t()))

                xgr = torch.mm(subspace, xg[i])
                xgr_list.append(xgr)

                with torch.no_grad():
                    running_subspace = self.__getattr__('running_subspace' + str(i))
                    running_subspace.data = (1 - self.momentum) * running_subspace.data + self.momentum * subspace.data

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu

            xr = torch.cat(xgr_list, dim=0)
            xr = xr * self.weight + self.bias
            xr = xr.view(C, N, H, W).transpose(0, 1)

            return xr

        else:
            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            x = (x - self.running_mean)
            G = self.groups
            xg = list(torch.chunk(x, G, dim=0))
            for i in range(G):
                subspace = self.__getattr__('running_subspace' + str(i))
                xg[i] = torch.mm(subspace, xg[i])
            x = torch.cat(xg, dim=0)
            x = x * self.weight + self.bias
            x = x.view(C, N, H, W).transpose(0, 1)
            return x

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(ZCANormPI, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class ZCANormPIunstable(nn.Module):
    def __init__(self, num_features, groups=1, eps=1e-4, momentum=0.1, affine=True):
        super(ZCANormPIunstable, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.groups = groups
        self.weight = Parameter(torch.Tensor(num_features, 1))
        self.bias = Parameter(torch.Tensor(num_features, 1))
        self.power_layer = power_iteration_unstable.apply
        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = int(self.num_features / self.groups)
        for i in range(self.groups):
            self.register_buffer("running_subspace{}".format(i), torch.eye(length, length))
            for j in range(length):
                self.register_buffer('eigenvector{}-{}'.format(i, j), torch.ones(length, 1))

    def reset_running_stats(self):
            self.running_mean.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        if self.training:
            N, C, H, W = x.size()
            G = self.groups
            x = x.transpose(0, 1).contiguous().view(C, -1)
            mu = x.mean(1, keepdim=True)
            x = x - mu
            xxt = torch.mm(x, x.t()) / (N * H * W) + torch.eye(C, out=torch.empty_like(x)) * self.eps

            assert C % G == 0
            length = int(C/G)
            xxti = torch.chunk(xxt, G, dim=0)
            xxtj = [torch.chunk(xxti[j], G, dim=1)[j] for j in range(G)]

            xg = list(torch.chunk(x, G, dim=0))

            xgr_list = []
            for i in range(G):
                subspace = torch.zeros_like(xxtj[i])
                # with torch.no_grad():
                #     print(xxtj[i])
                #     u, e, v = torch.svd(xxtj[i])
                #     print('eigenvalues via svd: {}'.format(e))
                for j in range(length):
                    # initialize eigenvector with random values
                    eigenvector_ij = self.__getattr__('eigenvector{}-{}'.format(i, j))
                    v = l2normalize(torch.randn_like(eigenvector_ij))
                    eigenvector_ij.data = v.data

                    eigenvector_ij = self.power_layer(xxtj[i], eigenvector_ij)
                    # eigenvector_ij.register_hook(print)
                    lambda_current = torch.mm(xxtj[i].mm(eigenvector_ij).t(), eigenvector_ij)/torch.mm(eigenvector_ij.t(), eigenvector_ij)
                    if j == 0:
                        lambda_ij = lambda_current
                    elif lambda_ij < lambda_current or lambda_current < self.eps:
                        # print('lambda_group{}_{}: current:{} previous:{}'.format(i, j, lambda_ij.item(), lambda_current.item()))
                        break
                    else:
                        lambda_ij = lambda_current
                    subspace += torch.mm(eigenvector_ij, torch.rsqrt(lambda_ij).mm(eigenvector_ij.t()))
                    # remove projections on the eigenvectors
                    xxtj[i] = xxtj[i] - torch.mm(xxtj[i], eigenvector_ij.mm(eigenvector_ij.t()))

                xgr = torch.mm(subspace, xg[i])
                xgr_list.append(xgr)

                with torch.no_grad():
                    running_subspace = self.__getattr__('running_subspace' + str(i))
                    running_subspace.data = (1 - self.momentum) * running_subspace.data + self.momentum * subspace.data

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu

            xr = torch.cat(xgr_list, dim=0)
            xr = xr * self.weight + self.bias
            xr = xr.view(C, N, H, W).transpose(0, 1)

            return xr

        else:
            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            x = (x - self.running_mean)
            G = self.groups
            xg = list(torch.chunk(x, G, dim=0))
            for i in range(G):
                subspace = self.__getattr__('running_subspace' + str(i))
                xg[i] = torch.mm(subspace, xg[i])
            x = torch.cat(xg, dim=0)
            x = x * self.weight + self.bias
            x = x.view(C, N, H, W).transpose(0, 1)
            return x

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(ZCANormPIunstable, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class ZCANormPI_debug(nn.Module):
    def __init__(self, num_features, groups=8, eps=1e-4, momentum=0.1, affine=True):
        super(ZCANormPI_debug, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.groups = groups
        self.weight = Parameter(torch.Tensor(num_features, 1))
        self.bias = Parameter(torch.Tensor(num_features, 1))
        self.power_layer = power_iteration_unstable.apply
        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = int(self.num_features / self.groups)
        for i in range(self.groups):
            self.register_buffer("running_subspace{}".format(i), torch.eye(length, length))
            for j in range(length):
                self.register_buffer('eigenvector{}-{}'.format(i, j), torch.ones(length, 1))

    def reset_running_stats(self):
            self.running_mean.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        if self.training:
            N, C, H, W = x.size()
            G = self.groups
            x = x.transpose(0, 1).contiguous().view(C, -1)
            mu = x.mean(1, keepdim=True)
            x = x - mu
            xxt = torch.mm(x, x.t()) / (N * H * W) + torch.eye(C, out=torch.empty_like(x)) * self.eps

            assert C % G == 0
            length = int(C/G)
            xxti = torch.chunk(xxt, G, dim=0)
            xxtj = [torch.chunk(xxti[j], G, dim=1)[j] for j in range(G)]

            xg = list(torch.chunk(x, G, dim=0))

            xgr_list = []
            for i in range(G):
                subspace = torch.zeros_like(xxtj[i])
                print('mat: ', xxtj[i])
                tmp_v = []

                with torch.no_grad():
                    u, e, v = torch.svd(xxtj[i])
                    # print('eigenvalues via svd: {}'.format(e))
                for j in range(length):
                    # initialize eigenvector with random values
                    eigenvector_ij = self.__getattr__('eigenvector{}-{}'.format(i, j))
                    # if j <= 0:
                    #     eigenvector_ij.data = v[:, j][..., None].data
                    # else:
                    #     vrand = l2normalize(torch.randn_like(eigenvector_ij))
                    #     eigenvector_ij.data = vrand.data

                    vrand = l2normalize(torch.randn_like(eigenvector_ij))
                    print('vrand: ', vrand.t())
                    tmp_v.append(vrand)

                    eigenvector_ij.data = vrand.data
                    eigenvector_ij = self.power_layer(xxtj[i], eigenvector_ij)
                    # eps = 1e-06
                    if j>0:
                        print('PI eig-{} (angle) between current and dominant eigenvectors'.format(j),
                              F.cosine_similarity(eigenvector_ij, v[:, 0][..., None], dim=0).clamp(-1.0, 1.0).abs().acos().item()/math.pi*180
                              )
                        print('SVD eig-{} (angle) between current and dominant eigenvectors'.format(j),
                              F.cosine_similarity(v[:, j][..., None], v[:, 0][..., None], dim=0).clamp(-1.0, 1.0).abs().acos().item() / math.pi * 180
                              )
                    # eigenvector_ij.register_hook(print)
                    lambda_ij = torch.mm(xxtj[i].mm(eigenvector_ij).t(), eigenvector_ij)/torch.mm(eigenvector_ij.t(), eigenvector_ij)
                    print('lambda_group{}_{}: {} vs gt {}'.format(i, j, lambda_ij.item(), e[j].item()))
                    subspace += torch.mm(eigenvector_ij, torch.rsqrt(lambda_ij).mm(eigenvector_ij.t()))
                    # remove projections on the eigenvectors
                    xxtj[i] = xxtj[i] - torch.mm(xxtj[i], eigenvector_ij.mm(eigenvector_ij.t()))
                    if lambda_ij<0:
                        import IPython
                        IPython.embed()

                xgr = torch.mm(subspace, xg[i])
                xgr_list.append(xgr)

                with torch.no_grad():
                    running_subspace = self.__getattr__('running_subspace' + str(i))
                    running_subspace.data = (1 - self.momentum) * running_subspace.data + self.momentum * subspace.data

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu

            xr = torch.cat(xgr_list, dim=0)
            xr = xr * self.weight + self.bias
            xr = xr.view(C, N, H, W).transpose(0, 1)

            return xr

        else:
            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            x = (x - self.running_mean)
            G = self.groups
            xg = list(torch.chunk(x, G, dim=0))
            for i in range(G):
                subspace = self.__getattr__('running_subspace' + str(i))
                xg[i] = torch.mm(subspace, xg[i])
            x = torch.cat(xg, dim=0)
            x = x * self.weight + self.bias
            x = x.view(C, N, H, W).transpose(0, 1)
            return x

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        super(ZCANormPI_debug, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)