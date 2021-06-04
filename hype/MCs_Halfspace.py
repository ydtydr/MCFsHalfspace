#!/usr/bin/env python3
import torch as th
from torch.autograd import Function
from .common import acosh, Renormalize, Grow_Exp, Scaling, ScalingV, Addmc
from .manifold import Manifold

factor_=20
class MCsHalfspaceManifold(Manifold):
    __slots__ = ["eps", "_eps", "norm_clip", "max_norm", "debug", "com_n"]

    @staticmethod
    def dim(dim):
        return dim

    def __init__(self, eps=1e-12, _eps=1e-5, norm_clip=1, max_norm=1e6,
            debug=False, com_n=1, **kwargs):
        self.eps = eps
        self._eps = _eps
        self.norm_clip = norm_clip
        self.max_norm = max_norm
        self.debug = debug
        self.com_n = com_n
    
    def sinhc(self, u):
        return th.div(th.sinh(u), u)

    def from_poincare_ball(self, point):
        u = point[..., :-1]
        v = point[..., -1]
        x = 2.0 / (u.norm(dim=-1).pow(2) + (v - 1.0).pow(2)).unsqueeze(-1) * u
        y = (1.0 - u.norm(dim=-1).pow(2) - v.pow(2)) / (u.norm(dim=-1).pow(2) + (v - 1.0).pow(2))
        return th.cat([x, y.unsqueeze(-1)], dim=-1)

    def to_poincare_ball(self, point):
        d = (point.size(-1)-1)//self.com_n+1
        x = point[..., :(d-1)]
        y = point[..., -self.com_n]
        for i in range(1,self.com_n,1):
            x += point[..., i*(d-1):(i+1)*(d-1)]
            y += point[..., -self.com_n+i]
        y = point[..., -1]
        u = 2.0/(x.norm(dim=-1).pow(2) + (y + 1.0).pow(2)).unsqueeze(-1) * x
        v = (x.norm(dim=-1).pow(2) + y.pow(2) - 1.0) / (x.norm(dim=-1).pow(2) + (y + 1.0).pow(2))
        return th.cat([u, v.unsqueeze(-1)], dim=-1)

    def distance(self, u, v):
        dis = MCsHalfspaceDistance.apply(u, v, self.com_n)
        return dis

    def pnorm(self, u):
        return th.sqrt(th.sum(th.pow(self.to_poincare_ball(u), 2), dim=-1))

    def normalize(self, w):
        """Normalize vector such that it is located on the halfspace model"""
        d = w.size(-1)//self.com_n
        narrowed = w.narrow(-1, -self.com_n, 1)
        narrowed.clamp_(min=0.0)
        return w

    def init_weights(self, w, irange=1e-5):        
        th.manual_seed(42)
        w.data.zero_()
        d = w.size(-1)//self.com_n
        ############
#         w.data[...,:d-1].uniform_(-irange, irange)
#         w.data[...,-self.com_n] = 1.0 + irange * 2 * (th.rand_like(w[...,-1])-0.5)
        ############
        iw = th.load('./init_embedding/mm_h2d_s1_v40.pt').data
        w.data[...,:d-1].copy_(self.normalize(iw.data)[...,:d-1])
        w.data[...,-self.com_n].copy_(self.normalize(iw.data)[...,-1])

    def rgrad(self, p, d_p):
        """Euclidean gradient for halfspace"""
        if d_p.is_sparse:
            u = d_p._values()
            x = p.index_select(0, d_p._indices().squeeze())
        else:
            u = d_p
            x = p
        d = x.size(-1)//self.com_n
        xn = x[..., -self.com_n]
        for i in range(1, self.com_n,1):
            xn += x[..., -self.com_n+i]
        assert xn.min()>0.0
        u.mul_(xn.unsqueeze(-1))### transform from Euclidean grad to Riemannian grad
        return d_p

    def expm(self, p, d_p, lr=None, out=None, normalize=False):
        """Exponential map for halfspace model"""
        d = p.size(-1)//self.com_n
        if out is None:
            out = p
        if d_p.is_sparse:
            ix, d_val_all = d_p._indices().squeeze(), d_p._values()
            d_val = d_val_all[...,:d]
            p_val = self.normalize(p.index_select(0, ix))
            xn = p_val[..., -self.com_n]
            for i in range(1, self.com_n, 1):
                xn += p_val[..., -self.com_n+i]
            assert xn.min()>0.0
            newp_val = p_val.clone()
            ################## Original Exponential map arithmetic
#             s = th.norm(d_val,dim=-1)#n
#             zeros_mask = (s == 0.0)
#             coshs = th.cosh(s)
#             sinhcs = self.sinhc(s)
#             sinhcs[zeros_mask] = th.ones_like(sinhcs[th.isnan(sinhcs)])
# #             sinhcs[th.isnan(sinhcs)] = th.ones_like(sinhcs[th.isnan(sinhcs)])
#             assert th.isnan(coshs).max()==0, "coshs includes NaNs"
#             assert th.isnan(sinhcs).max()==0, "sinhcs includes NaNs"
#             grad_f_x = th.div(xn, th.div(coshs, sinhcs)-d_val[...,-1]).unsqueeze(-1) * d_val[...,:-1]#n*(d-1)
#             growed_x = Grow_Exp(p_val[...,:-self.com_n], grad_f_x, d-1, self.com_n)#n*(m+1)(d-1)
#             newp_val[...,:-self.com_n] = Renormalize(growed_x, d-1, self.com_n+1)#n*m(d-1)
#             scaled_y = Scaling(p_val[..., -self.com_n:], th.div(1.0, coshs-d_val[...,-1]*sinhcs))#n*(m+1)
#             assert th.isnan(scaled_y).max()==0, "scaled_y includes NaNs"
#             newp_val[...,-self.com_n:] = Renormalize(scaled_y, 1, self.com_n+1)#n*m(d-1)
#             assert th.isnan(newp_val[...,-self.com_n:]).max()==0, "newp_val[...,-self.com_n:] includes NaNs"
            #################### Numerical stable form of the exponential map
            mask_pos = d_val[...,-1]>0
            mask_neg = d_val[...,-1]<=0
            Pos = d_val[mask_pos]
            Neg = d_val[mask_neg]
            if len(Pos) !=0:
                s_postive = th.norm(Pos, dim=-1)
                r_square = th.sum(th.pow(Pos[...,:-1], 2), dim=-1)
                scoths = th.div(s_postive, th.tanh(s_postive))
                scschs_square = th.pow(th.div(s_postive, th.sinh(s_postive)),2)
                grad_f_x_pos = (th.div(scoths + Pos[..., -1], scschs_square + r_square) * xn[mask_pos]).unsqueeze(-1) * Pos[...,:-1]
                growed_x_pos = Grow_Exp(p_val[...,:-self.com_n][mask_pos], grad_f_x_pos, d-1, self.com_n)#n*(m+1)(d-1)
                newp_val[..., :-self.com_n][mask_pos] = growed_x_pos[...,:self.com_n*(d-1)]#n*m(d-1)
#                 newp_val[..., :-self.com_n][mask_pos] = Renormalize(growed_x_pos, d-1, self.com_n+1)#n*m(d-1)
                scaled_y_pos = ScalingV(p_val[..., -self.com_n:][mask_pos], th.div(s_postive + Pos[..., -1] * th.tanh(s_postive), r_square + th.pow(th.div(Pos[..., -1], th.cosh(s_postive)) ,2), ) * th.div(s_postive, th.cosh(s_postive)))#n*(m+1)
                newp_val[...,-self.com_n:][mask_pos] = Renormalize(scaled_y_pos, 1, self.com_n+1)#n*m(d-1)  
            if len(Neg) !=0:
                s_negative = th.norm(Neg, dim=-1)
                zeros_mask = (s_negative == 0.0)
                coshs = th.cosh(s_negative)
                sihncs = self.sinhc(s_negative)
                sihncs[zeros_mask] = th.ones_like(sihncs[th.isnan(sihncs)])
                grad_f_x_neg = th.div(xn[mask_neg], th.div(coshs, sihncs)-Neg[...,-1]).unsqueeze(-1) * Neg[...,:-1]#n*(d-1)
                growed_x_neg = Grow_Exp(p_val[...,:-self.com_n][mask_neg], grad_f_x_neg, d-1, self.com_n)#n*(m+1)(d-1)
                newp_val[..., :-self.com_n][mask_neg] = growed_x_neg[...,:self.com_n*(d-1)]
#                 newp_val[..., :-self.com_n][mask_neg] = Renormalize(growed_x_neg, d-1, self.com_n+1)#n*m(d-1)
                scaled_y_neg = ScalingV(p_val[..., -self.com_n:][mask_neg], th.div(1.0, coshs-Neg[...,-1]*sihncs))#n*(m+1)
                newp_val[...,-self.com_n:][mask_neg] = Renormalize(scaled_y_neg, 1, self.com_n+1)#n*m(d-1)
            ##################
            newp_val = self.normalize(newp_val)
            assert newp_val[...,-self.com_n].min()>0.0
            p.index_copy_(0, ix, newp_val)
        else:
            raise NotImplementedError

class MCsHalfspaceDistance(Function):
    @staticmethod
    def forward(self, u, v, com_n, myeps = 0.0):
        self.com_n = com_n
        assert th.isnan(u).max()==0, "u includes NaNs"
        assert th.isnan(v).max()==0, "v includes NaNs"
        if len(u)<len(v):
            u = u.expand_as(v)
        elif len(u)>len(v):
            v = v.expand_as(u)
        d = u.size(-1)//com_n
        self.save_for_backward(u, v)
        ##########
        u_v_x_o = Addmc(u[...,:-self.com_n], -v[...,:-self.com_n], d-1, self.com_n)
        u_v_y_o = Addmc(u[...,-self.com_n:], -v[...,-self.com_n:], 1, self.com_n)
        u_v_x = u_v_x_o[...,:d-1]
        u_v_y = u_v_y_o[...,0]
        self.un = u[..., -self.com_n]
        self.vn = v[..., -self.com_n]
        for i in range(1, com_n, 1):
            u_v_x += u_v_x_o[...,i*(d-1):(i+1)*(d-1)]
            u_v_y += u_v_y_o[...,i]#m*n*(d-1)
            self.un += u[..., -self.com_n+i]
            self.vn += v[..., -self.com_n+i]
        #########
        self.u_v = th.cat([u_v_x, u_v_y.unsqueeze(-1)],-1)
        assert self.un.min()>0.0
        assert self.vn.min()>0.0
        sqnorm = th.sum(th.pow(self.u_v, 2), dim=-1)#m*n
        self.x_ = th.div(sqnorm, 2*self.un*self.vn)#m*n
        self.z_ = th.sqrt(self.x_ * (self.x_ + 2))#m*n
        return th.log1p(self.x_ + self.z_)

    @staticmethod
    def backward(self, g):
        u, v = self.saved_tensors ## u, v are of size b*neg*(com_n*d)
        self.z_[self.z_==0.0] = th.ones_like(self.z_[self.z_==0.0]) * 1e-6
        d = u.size(-1)//self.com_n
        g = g.unsqueeze(-1).expand_as(u)
        gu = th.zeros_like(u)  # b*neg*(com_n*d)
        gv = th.zeros_like(v)  # b*neg*(com_n*d)
        auxli_term = th.div(self.u_v, (self.un * self.vn).unsqueeze(-1))#b*neg*(d)
        gu[..., :d-1] = th.div(1, self.z_).unsqueeze(-1) * auxli_term[...,:d-1]#b*neg*(d-1)
        gu[..., d-1] = th.div(1, self.z_) * (auxli_term[...,-1] - th.div(self.x_, self.un))#b*neg
        gv[..., :d-1] = -1 * gu[..., :d-1]  # #b*neg*(d-1)
        gv[..., d-1] = th.div(1, self.z_) * (-1 * auxli_term[...,-1] - th.div(self.x_, self.vn))#b*neg
        assert th.isnan(gu).max() == 0, "gu includes NaNs"
        assert th.isnan(gv).max() == 0, "gv includes NaNs"        
        return g * gu, g * gv, None