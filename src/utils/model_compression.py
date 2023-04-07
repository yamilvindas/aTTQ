#!/usr/bin/env python3
"""
    Implementation of some functions useful for model compression using ternary
    quantization
"""
import torch

#==============================================================================#
#====Functions for Ternary Networks from the paper of Heinrich et al. (2018)====#
#==============================================================================#
# ternary weight approximation according to https://arxiv.org/abs/1605.04711
def approx_weights(w_in):
    """
        Function from https://github.com/mattiaspaul/TernaryNet/blob/master/ternaryNet_github.py
    """
    a,b,c,d = w_in.size()
    delta = 0.7*torch.mean(torch.mean(torch.mean(torch.abs(w_in),dim=3),dim=2),dim=1).view(-1,1,1,1)
    alpha = torch.abs(w_in)*(torch.abs(w_in)>delta).float()
    alpha = (torch.sum(torch.sum(torch.sum(alpha,dim=3),dim=2),dim=1)  \
    /torch.sum(torch.sum(torch.sum((alpha>0).float(),dim=3),dim=2),dim=1)).view(-1,1,1,1)
    w_out = -(w_in<-delta).float()*alpha + (w_in>delta).float()*alpha
    return w_out

# ternary weight approximation for FC layers
def approx_weights_fc(w_in):
    delta = 0.7*torch.mean(torch.abs(w_in),dim=1).view(-1,1)
    alpha = torch.abs(w_in)*(torch.abs(w_in)>delta).float()
    alpha = (torch.sum(alpha,dim=1)  \
    /torch.sum((alpha>0).float(),dim=1)).view(-1,1)
    w_out = -(w_in<-delta).float()*alpha + (w_in>delta).float()*alpha
    return w_out


#==============================================================================#
#=====Functions for asymmetric ternary quantization from Zhu et al. (2017)=====#
#==============================================================================#
def quantize(kernel, w_p, w_n, t):
    """
    Function from: https://github.com/TropComplique/trained-ternary-quantization/blob/master/utils/quantization.py

    Return quantized weights of a layer.
    Only possible values of quantized weights are: {zero, w_p, -w_n}.
    """
    delta = t*kernel.abs().max()
    a = (kernel > delta).float()
    b = (kernel < -delta).float()
    return w_p*a + (-w_n*b)

def quantize_two_thresh(kernel, w_r, w_l, x, y):
    """
    Function based on: https://github.com/TropComplique/trained-ternary-quantization/blob/master/utils/quantization.py
    ATTENTION: it is not the same function as we change the method to quantize
    the weights.

    Return quantized weights of a layer.
    Only possible values of quantized weights are: {zero, w_l, w_r}.
    """
    delta_min = kernel.mean() + x*kernel.std()
    delta_max = kernel.mean() + y*kernel.std()
    a = (kernel > delta_max).float()
    b = (kernel < delta_min).float()
    return w_r*a + w_l*b


def get_grads(kernel_grad, kernel, w_p, w_n, t):
    """
    Function from: https://github.com/TropComplique/trained-ternary-quantization/blob/master/utils/quantization.py

    Arguments:
        kernel_grad: gradient with respect to quantized kernel.
        kernel: corresponding full precision kernel.
        w_p, w_n: scaling factors.
        t: hyperparameter for quantization.
    Returns:
        1. gradient for the full precision kernel.
        2. gradient for w_p.
        3. gradient for w_n.
    """
    delta = t*kernel.abs().max()
    # masks
    a = (kernel > delta).float()
    b = (kernel < -delta).float()
    c = torch.ones(kernel.size()).cuda() - a - b
    # scaled kernel grad and grads for scaling factors (w_p, w_n)
    return w_p*a*kernel_grad + w_n*b*kernel_grad + 1.0*c*kernel_grad,\
        (a*kernel_grad).sum(), (b*kernel_grad).sum()

def get_grads_two_thresh(kernel_grad, kernel, w_r, w_l, x, y):
    """
    Function from: https://github.com/TropComplique/trained-ternary-quantization/blob/master/utils/quantization.py
    ATTENTION: it is not the same function as we change the method to quantize
    the weights.

    Arguments:
        kernel_grad: gradient with respect to quantized kernel.
        kernel: corresponding full precision kernel.
        w_r, w_l: scaling factors.
        x, y: hyperparameter for quantization.
    Returns:
        1. gradient for the full precision kernel.
        2. gradient for w_r.
        3. gradient for w_l.
    """
    delta_min = kernel.mean() + x*kernel.std()
    delta_max = kernel.mean() + y*kernel.std()
    # masks
    a = (kernel > delta_max).float()
    b = (kernel < delta_min).float()
    c = torch.ones(kernel.size()).cuda() - a - b
    # scaled kernel grad and grads for scaling factors (w_p, w_n)
    return w_r*a*kernel_grad + w_l*b*kernel_grad + 1.0*c*kernel_grad,\
        (a*kernel_grad).sum(), (b*kernel_grad).sum()
