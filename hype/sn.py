#!/usr/bin/env python3

import torch as th
from torch import nn
from numpy.random import randint
from . import graph
from .graph_dataset import BatchedDataset

model_name = '%s_dim%d%com_n'


class Embedding(graph.Embedding):
    def __init__(self, size, dim, manifold, sparse=True, com_n=1):
        super(Embedding, self).__init__(size, dim, manifold, sparse, com_n)
        self.lossfn = nn.functional.cross_entropy
        self.manifold = manifold

    def _forward(self, e, int_matrix=None):
        o = e.narrow(1, 1, e.size(1) - 1)
        s = e.narrow(1, 0, 1).expand_as(o)###source
        if 'LTiling' in str(self.manifold):
            o_int_matrix = int_matrix.narrow(1, 1, e.size(1) - 1)
            s_int_matrix = int_matrix.narrow(1, 0, 1).expand_as(o_int_matrix)###source
            dists = self.dist(s, s_int_matrix, o, o_int_matrix).squeeze(-1)
        else:
            dists = self.dist(s, o).squeeze(-1)
        return -dists
    
    def loss(self, preds, targets, weight=None, size_average=True):
        return self.lossfn(preds, targets)


# This class is now deprecated in favor of BatchedDataset (graph_dataset.pyx)
class Dataset(graph.Dataset):
    def __getitem__(self, i):
        t, h = self.idx[i]
        negs = set()
        ntries = 0
        nnegs = int(self.nnegatives())
        if t not in self._weights:
            negs.add(t)
#             print(negs)
        else:
            while ntries < self.max_tries and len(negs) < nnegs:
                if self.burnin:
                    n = randint(0, len(self.unigram_table))
                    n = int(self.unigram_table[n])
                else:
                    n = randint(0, len(self.objects))
                if (n not in self._weights[t]) or \
                        (self._weights[t][n] < self._weights[t][h]):
                    negs.add(n)
                ntries += 1
        if len(negs) == 0:
            negs.add(t)
        ix = [t, h] + list(negs)
        while len(ix) < nnegs + 2:
            ix.append(ix[randint(2, len(ix))])
#         print(ix)
#         assert 1==2
        return th.LongTensor(ix).view(1, len(ix)), th.zeros(1).long()


def initialize(manifold, opt, idx, objects, weights, sparse=True):
    conf = []
    mname = model_name % (opt.manifold, opt.dim, opt.com_n)
    data = BatchedDataset(idx, objects, weights, opt.negs, opt.batchsize,
        opt.ndproc, opt.burnin > 0, opt.dampening)
    model = Embedding(
        len(data.objects),
        opt.dim,
        manifold,
        sparse=sparse,
        com_n=opt.com_n,
    )
    data.objects = objects
    return model, data, mname, conf
