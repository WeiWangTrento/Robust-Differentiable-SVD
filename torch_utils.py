import torch as th
# from torch.nn.functional import normalize
import torch.nn as nn
from time import sleep
import time
'''
This file contains all the custom pytorch operator.
'''


# class power_iteration(nn.Module):
#     def __init__(self, n_power_iterations=2, eps=1e-5):
#         super(power_iteration, self).__init__()
#         self.n_power_iterations = n_power_iterations
#         self.eps = eps
#         self.power_operation = power_iteration_once.apply
#
#     def _check_input_shape(self, M, v):
#         if M.shape[0] != M.shape[1] or M.dim() != 2:
#             raise ValueError('2D covariance matrix size is {}, but it should be square'.format(M.shape()))
#         if M.shape[0] != v.shape[0]:
#             raise ValueError('input covariance dim {} should equal to eig-vector shape {})'.
#                              format(M.shape[0], v.shape[0]))
#
#     def forward(self, M, v):
#         self._check_input_shape(M, v)
#         for k in range(self.n_power_iterations):
#             v = self.power_operation(M, v)
#         return v


class power_iteration_unstable(th.autograd.Function):
    @staticmethod
    def forward(ctx, M, v_k, num_iter=19):
        '''
        :param ctx: used to save meterials for backward.
        :param M: n by n matrix.
        :param v_k: initial guess of leading vector.
        :return: v_k1 leading vector.
        '''
        vk_list = []
        vk_list.append(v_k)
        ctx.num_iter = num_iter
        for _ in range(int(ctx.num_iter)):
            v_k = M.mm(v_k)
            v_k /= th.norm(v_k).clamp(min=1.e-5)
            vk_list.append(v_k)

        ctx.save_for_backward(M, *vk_list)
        return v_k

    @staticmethod
    def backward(ctx, grad_output):
        M = ctx.saved_tensors[0]
        vk_list = ctx.saved_tensors[1:]
        dL_dvk1 = grad_output
        dL_dM = 0
        # print('befor. mat-bp gradients {}'.format(dL_dvk1))
        # print('befor. mat-bp gradients {0:0.6f}'.format(dL_dvk1.abs().max().item()))
        for i in range(1, ctx.num_iter + 1):
            v_k1 = vk_list[-i]
            v_k = vk_list[-i - 1]
            mid = calc_mid(M, v_k, v_k1, dL_dvk1)
            dL_dM += mid.mm(th.t(v_k))
            dL_dvk1 = M.mm(mid)
        # print('after. mat-bp gradients {0:0.6f}'.format(dL_dM.abs().max().item()))
        return dL_dM, dL_dvk1


def calc_mid(M, v_k, v_k1, dL_dvk1):
    I = th.eye(M.shape[-1], out=th.empty_like(M))
    mid = (I - v_k1.mm(th.t(v_k1)))/th.norm(M.mm(v_k)).clamp(min=1.e-5)
    mid = mid.mm(dL_dvk1)
    return mid


class power_iteration_once(th.autograd.Function):
    @staticmethod
    def forward(ctx, M, v_k, num_iter=9):
        '''
        :param ctx: used to save meterials for backward.
        :param M: n by n matrix.
        :param v_k: initial guess of leading vector.
        :return: v_k1 leading vector.
        '''
        ctx.num_iter = num_iter
        ctx.save_for_backward(M, v_k)
        return v_k

    @staticmethod
    def backward(ctx, grad_output):
        M, v_k = ctx.saved_tensors
        dL_dvk = grad_output
        I = th.eye(M.shape[-1], out=th.empty_like(M))
        numerator = I - v_k.mm(th.t(v_k))
        denominator = th.norm(M.mm(v_k)).clamp(min=1.e-5)
        ak = numerator / denominator
        term1 = ak
        q = M / denominator
        for i in range(1, ctx.num_iter + 1):
            ak = q.mm(ak)
            term1 += ak
        dL_dM = th.mm(term1.mm(dL_dvk), v_k.t())
        return dL_dM, ak


def bug_geometric_approximation(s):
    I = s.new(th.eye(s.shape[0]).numpy())
    p = s[..., None] / s[None] - I
    p = th.where(p < 1., p, 1. / p)
    a1 = s.repeat(s.shape[0], 1).t()
    a1_t = a1.t()
    a1 = 1. / th.where(a1 >= a1_t, a1, - a1_t)
    a1 *= a1.new(th.ones(s.shape[0], s.shape[0]).numpy()) - I
    p_hat = th.ones_like(p)
    for i in range(9):
        p_hat = p_hat * p
        a1 += a1 * p_hat
    return a1


def geometric_approximation(s):
    dtype = s.dtype
    I = th.eye(s.shape[0], device=s.device).type(dtype)
    p = s.unsqueeze(-1) / s.unsqueeze(-2) - I
    p = th.where(p < 1., p, 1. / p)
    a1 = s.repeat(s.shape[0], 1).t()
    a1_t = a1.t()
    a1 = 1. / th.where(a1 >= a1_t, a1, - a1_t)
    a1 *= th.ones(s.shape[0], s.shape[0], device=s.device).type(dtype) - I
    p_app = th.ones_like(p)
    p_hat = th.ones_like(p)
    for i in range(9):
        p_hat = p_hat * p
        p_app += p_hat
    a1 = a1 * p_app
    return a1


def geometric_approximation_old(s):
    dtype = s.dtype
    I = th.eye(s.shape[0], device=s.device).type(dtype)
    p = s[..., None] / s[None] - I
    p = th.where(p < 1., p, 1. / p)
    a1 = s.repeat(s.shape[0], 1).t()
    a1_t = a1.t()
    a1 = 1. / th.where(a1 >= a1_t, a1, - a1_t)
    a1 *= th.ones(s.shape[0], s.shape[0], device=s.device).type(dtype) - I
    for i in range(9):
        p = p * p
        a1 += a1 * p
    return a1


def clip(s):
    dtype = s.dtype
    a1 = s.repeat(s.shape[0], 1).t()
    a1_t = a1.t()
    diff = a1 - a1_t
    diff = diff.clamp(min=0)
    diff = th.where(diff>0, diff.clamp(min=0.01), diff)
    diff[diff>0] = 1./diff[diff>0]
    diff = diff - diff.t()
    return diff


def v3geometric_approximation(s, eps=1e-2):
    dtype = s.dtype
    I = th.eye(s.shape[0], device=s.device).type(dtype)
    p = s.unsqueeze(-1) / s.unsqueeze(-2) - I
    p = th.where(p < 1., p, 1. / p)
    a1 = s.repeat(s.shape[0], 1).t()
    a1_t = a1.t()
    a1 = 1. / th.where(a1 >= a1_t, a1, - a1_t)
    a1 *= th.ones(s.shape[0], s.shape[0], device=s.device).type(dtype) - I
    a1 = th.where(a1.abs() >= 1/eps, th.zeros_like(a1), a1)
    p[a1.abs() == 0] = 0
    # print('0 value number', (p.abs() == 0).sum()-64)
    p_acc = th.ones_like(p) + p
    for i in range(2):
        p = p * p
        p_acc += p_acc * p
    a1 = a1 * p_acc
    return a1


class svdv2(th.autograd.Function):
    @staticmethod
    def forward(ctx, M):
        # s, u = th.symeig(M, eigenvectors=True, upper=True)  # s in a ascending sequence.
        ut, s, u = th.svd(M)  # s in a descending sequence.
        # print('0-eigenvalue # {}'.format((s <= 1e-5).sum()))
        s = th.clamp(s, min=1e-10)  # 1e-5
        ctx.save_for_backward(M, u, s)
        return u, s

    @staticmethod
    def backward(ctx, dL_du, dL_ds):
        M, u, s = ctx.saved_tensors
        # I = th.eye(s.shape[0])
        # K_t = 1.0 / (s[None] - s[..., None] + I) - I
        K_t = geometric_approximation(s).t()
        u_t = u.t()
        dL_dM = u.mm(K_t * u_t.mm(dL_du) + th.diag(dL_ds)).mm(u_t)
        return dL_dM

class svdv2clip(th.autograd.Function):
    @staticmethod
    def forward(ctx, M):
        # s, u = th.symeig(M, eigenvectors=True, upper=True)  # s in a ascending sequence.
        ut, s, u = th.svd(M)  # s in a descending sequence.
        # print('0-eigenvalue # {}'.format((s <= 1e-5).sum()))
        s = th.clamp(s, min=1e-5)
        ctx.save_for_backward(M, u, s)
        return u, s

    @staticmethod
    def backward(ctx, dL_du, dL_ds):
        M, u, s = ctx.saved_tensors
        # I = th.eye(s.shape[0])
        # K_t = 1.0 / (s[None] - s[..., None] + I) - I
        K_t = clip(s).t()
        u_t = u.t()
        dL_dM = u.mm(K_t * u_t.mm(dL_du) + th.diag(dL_ds)).mm(u_t)
        return dL_dM