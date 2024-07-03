import torch
import numpy as np


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor


class SinkhornKnopp(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_iters = args.num_iters_sk
        self.epsilon = args.epsilon_sk

    @torch.no_grad()
    def iterate(self, Q):
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / Q.shape[1]  # Samples

        # obtain permutation/order from the marginals
        marginals_argsort = torch.argsort(Q.sum(1))
        marginals_argsort = marginals_argsort.detach()
        r = []
        for i in range(Q.shape[0]):  # Classes
            r.append((1 / 1) ** (i / (Q.shape[0] - 1.0)))
        r = np.array(r)
        r = r * (Q.shape[1] / Q.shape[0])  # Per-class distribution in the mini-batch
        r = torch.from_numpy(r).cuda(non_blocking=True)
        r[marginals_argsort] = torch.sort(r)[0]  # Sort/permute based on the data order
        r = torch.clamp(r, min=1)  # Clamp the min to have a balance distribution for the tail classes
        r /= r.sum()  # Scaling to make it prob

        for it in range(self.num_iters):
            u = torch.sum(Q, dim=1)
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    @torch.no_grad()
    def forward(self, logits):
        # get assignments
        # import pdb; pdb.set_trace()
        q = logits / self.epsilon
        M = torch.max(q)
        q -= M
        q = torch.exp(q).t()
        return self.iterate(q)
