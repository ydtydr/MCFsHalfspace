#!/usr/bin/env python3
import torch as th
from torch.autograd import Function


class Acosh(Function):
    @staticmethod
    def forward(ctx, x, eps):
        z = th.sqrt(x * x - 1)
        ctx.save_for_backward(z)
        ctx.eps = eps
        return th.log(x + z)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        z = th.clamp(z, min=ctx.eps)
        z = g / z
        return z, None


acosh = Acosh.apply

def Flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = th.arange(x.size(dim) - 1, -1, -1,
                                dtype=th.long, device=x.device)
    return x[tuple(indices)]

def Two_Sum(a, b):
    """
    performs two-sum addition of two standard float vectors
    """
    x = a + b
    b_virtual = x - a
    a_virtual = x - b_virtual
    b_roundoff = b - b_virtual
    a_roundoff = a - a_virtual
    y = a_roundoff + b_roundoff
    return x, y

def QTwo_Sum(a, b):
    """
    performs quick-two-sum addition of two standard float vectors, work for n dimensional vectors, and b*n dimension
    """
    s = a + b
    e = b - (s - a)
    return s, e

def Split(a):
    """
    The following algorithm splits a 53-bit IEEE double precision floating point number into ahi and alo, each with 26 bits of significand, such that a = ahi + alo. ahi will contain the first 26 bits, while alo will contain the lower 26 bits.
    """
    t = (2**26+1)*a
    ahi = t-(t-a)
    alo = a - ahi
    return ahi, alo

def Two_Prod(a,b):
    """
    performs two-prod algorithm, computes p = fl(a × b) and e = err(a × b).
    """
    if len(b.size()) < len(a.size()):
        b = b.unsqueeze(-1)
    p = a * b
    ahi, alo = Split(a)
    bhi, blo = Split(b)
    e = ((ahi*bhi-p)+ahi*blo+alo*bhi)+alo*blo
    return p, e

def Renormalize(x, n, m):
    """
    renormalize m components (decreasing magnitude) floats inputs (of size b*(m*n) to outputs (of size b*((m-1)*n)
    this is a complete version, including if condition in the algorithm
    """
    if len(x.size())==3:
        rough_x = []
#         print('x', x.size())
        s = x[..., (m-1)*n:] # b*n
#         print('s', s.size())
        for i in range(m-1, 0, -1):
            s, rough_x_val = Two_Sum(x[..., (i-1)*n:i*n], s)### b*n
            rough_x.append(rough_x_val) ## in increasing magnitude here
        normalized_x = th.zeros(x.size(0),x.size(1),(m+2)*n)
        if x.is_cuda:
            normalized_x = normalized_x.cuda()
        for i in range(1, m, 1):
            s, e = Two_Sum(s, rough_x[-i])### b*n
            nonzero_ind = th.nonzero(e).t()
            normalized_x[nonzero_ind[0],nonzero_ind[1],n*(i-1)+nonzero_ind[2]] = s[nonzero_ind[0],nonzero_ind[1],nonzero_ind[2]]
            s[nonzero_ind[0],nonzero_ind[1],nonzero_ind[2]] = e[nonzero_ind[0],nonzero_ind[1],nonzero_ind[2]]
        normalized_x[...,m*n:(m+1)*n] = s
        normalized_x[...,(m+1)*n:] = e
#         print('normalized_x', normalized_x, normalized_x.size())
#         assert False
        normalized_x = normalized_x.view(x.size(0),x.size(1),(m+2), n)
        first_ind = th.arange(x.size(0)).view(x.size(0),1).repeat(1,x.size(1)*(m+2)*n).view(-1)
#         second_ind = th.arange(x.size(1)).view(x.size(1),1).repeat(1,x.size(0)*(m+2)*n).view(-1)
        second_ind = th.arange(x.size(1)).view(x.size(1),1).repeat(1,(m+2)).view(-1).repeat(1,x.size(0)*n).view(-1)
#         print(second_ind.size(), second_ind.size())
#         print('normalized_x', normalized_x[0,0:3,:,0], normalized_x.size())
        third_ind = th.argsort(th.abs(normalized_x), dim=2, descending=True).view(-1)
        last_ind = th.cat(x.size(0)*x.size(1)*(m+2)*[th.arange(n)])
#         print(first_ind[:3*7])
#         print(second_ind[:3*7])
#         print(third_ind[:3*7])
#         print(last_ind[:3*7])
        normalized_x = normalized_x[first_ind,second_ind,third_ind, last_ind].view(x.size(0),x.size(1),(m+2)*n)
#         print('normalized_x', normalized_x, normalized_x.size())
#         assert False
        return normalized_x[...,:(m-1)*n]### b*(m-1)n  normalized_x##b*mn
    else:
        rough_x = []
        s = x[..., (m-1)*n:] # b*n
        for i in range(m-1, 0, -1):
            s, rough_x_val = Two_Sum(x[..., (i-1)*n:i*n], s)### b*n
            rough_x.append(rough_x_val) ## in increasing magnitude here
        normalized_x = th.zeros(x.size(0),(m+2)*n)
        if x.is_cuda:
            normalized_x = normalized_x.cuda()
        for i in range(1, m, 1):
            s, e = Two_Sum(s, rough_x[-i])### b*n
            nonzero_ind = th.nonzero(e).t()
            normalized_x[nonzero_ind[0],n*(i-1)+nonzero_ind[1]] = s[nonzero_ind[0],nonzero_ind[1]]
            s[nonzero_ind[0],nonzero_ind[1]] = e[nonzero_ind[0],nonzero_ind[1]]
        normalized_x[...,m*n:(m+1)*n] = s
        normalized_x[...,(m+1)*n:] = e
        normalized_x = normalized_x.view(x.size(0),(m+2), n)
        first_ind = th.arange(x.size(0)).view(x.size(0),1).repeat(1,(m+2)*n).view(-1)
        third_ind = th.cat(x.size(0)*(m+2)*[th.arange(n)])
        second_ind = th.argsort(th.abs(normalized_x), dim=1, descending=True).view(-1)
        normalized_x = normalized_x[first_ind,second_ind,third_ind].view(x.size(0),(m+2)*n)
        return normalized_x[...,:(m-1)*n]### b*(m-1)n  normalized_x##b*mn

# def Renormalize_3d(x, n, m):
#     return x[...,:(m-1)*n]### b*(m-1)n  normalized_x##b*mn

def Grow_Exp(x, bf, n, m):
    """
    Performs batched addition of m components floats and standard floats
    Parameters
    ----------
    x : (b, m*n) tensor
        b: batchsize, an array of m-components floats, in decreasing magnitude
    bf : (b, n) tensor
        Array of standard floats.
    n: embedding dimension
    m: number of components

    Returns
    -------
    h : (b, (m+1)*n) tensor
        (m+1)-components floats, in decreasing magnitude
    """
    e = bf
    h = []
    for i in range(m):
        hval, e = Two_Sum(x[..., i*n:(i+1)*n], e)### hval: b*n
        h.append(hval)
    h.append(e)
    return th.cat(h, dim=-1)### b*((m+1)n)

def Addmc(x, bf, n, m):
    """
    Performs expansion addition
    ----------
    x : (b, neg, m*n) tensor
        b: batchsize, an array of m-components floats, in decreasing magnitude
    bf : (b, neg, m*n) tensor
    n: embedding dimension
    m: number of components

    Returns
    -------
    h : (b, neg, (2m)*n) tensor
        (2m)-components floats, in decreasing magnitude
    """
    h_out = []
    h = x
    for i in range(0,m,1):
        fs = bf[...,(m-i-1)*n:(m-i)*n]
        h_new = Grow_Exp(h, fs, n, m)
        h_out.append(h_new[...,-n:])##smallest to largest
        h = h_new[...,:-n]
    if len(x.size())==3:
        h_out = Flip(th.cat(h_out, dim=-1).view(x.size(0), x.size(1), m, n), -2).view_as(x) ##largest to smallest
    else:
        h_out = Flip(th.cat(h_out, dim=-1).view(x.size(0), m, n), -2).view_as(x)
    return th.cat([h, h_out], dim=-1)

def Addmc_old(x, bf, n, m):
    """
    Performs batched mulplication of m components floats and standard floats, scaling of an expansion
    Parameters
    ----------
    x : (b, m) tensor
        b: batchsize, an array of m-components floats, in decreasing magnitude
    bf : (b) tensor
        Array of standard floats.
    n: embedding dimension
    m: number of components

    Returns
    -------
    h : (b, (m+1)*n) tensor
        (m+1)-components floats, in decreasing magnitude
    """
    h = []
    hval, e = Two_Sum(x[...,:n], bf[...,:n])
    h.append(hval)
    for i in range(1, m, 1):
        hval_pre, e1 = Two_Sum(x[..., i*n:(i+1)*n], bf[..., i*n:(i+1)*n])
        hval, e2 = Two_Sum(hval_pre, e)
        h.append(hval)
        e = e1 + e2
    h.append(e)
    return th.cat(h, dim=-1)

def Aggregate(u, d, com_n, keepdim=False):
    if keepdim:
        u1 = th.zeros_like(u)
    else:
        u1 = u[...,:d]
    for i in range(1, com_n, 1):
        u1[...,:d] += u[...,i*d:(i+1)*d]
    return u1

def ScalingV(x, bf):
    """
    Performs batched mulplication of m components floats and standard floats, scaling of an expansion
    Parameters
    ----------
    x : (b, m) tensor
        b: batchsize, an array of m-components floats, in decreasing magnitude
    bf : (b) tensor
        Array of standard floats.
    n: embedding dimension
    m: number of components

    Returns
    -------
    h : (b, m+1) tensor
        (m+1)-components floats, in decreasing magnitude
    """
    m = x.size(-1)
    h = []
    hval, e = Two_Prod(x[..., 0], bf)
    h.append(hval)
    for i in range(1, m, 1):
        hval_pre, e1 = Two_Prod(x[..., i], bf)
        hval, e2 = Two_Sum(hval_pre, e)
        h.append(hval)
        e = e1 + e2
    h.append(e)
    return th.stack(h, dim=-1)

def Scaling(x, bf, d, com_n):
    """
    Performs batched mulplication of m components floats and standard floats, scaling of an expansion
    Parameters
    ----------
    x : (b, m) tensor
        b: batchsize, an array of m-components floats, in decreasing magnitude
    bf : (b) tensor
        Array of standard floats.
    n: embedding dimension
    m: number of components

    Returns
    -------
    h : (b, m+1) tensor
        (m+1)-components floats, in decreasing magnitude
    """
    h = []
    hval, e = Two_Prod(x[..., :d], bf)
    h.append(hval)
    for i in range(1, com_n, 1):
        hval_pre, e1 = Two_Prod(x[..., i*d:(i+1)*d], bf)
        hval, e2 = Two_Sum(hval_pre, e)
        h.append(hval)
        e = e1 + e2
    h.append(e)
    return th.cat(h, dim=-1)

# def ScalingMCs(x, bf, d, com_n):
#     returned_x = th.zeros(x.size(0), x.size(1), (com_n+1)*d)
#     if x.is_cuda:
#         returned_x = returned_x.cuda()
#     for i in range(d):
        



