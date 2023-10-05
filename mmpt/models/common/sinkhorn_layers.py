import torch
import torch.nn as nn
from PIL.Image import NONE
from ..registry import OPERATORS

@OPERATORS.register_module()
class Sinkhorn_Layers(nn.Module):
    def __init__(self, iters=3, init_bin_score=-1):
        super().__init__()
        self.iters = iters
        self.bin_score = nn.Parameter(
                torch.tensor(init_bin_score, requires_grad=True)) if init_bin_score != -1  else -1
        self.skh_iters = iters

    def log_sinkhorn_iterations(self, Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
        """ Perform Sinkhorn Normalization in Log-space for stability"""
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(iters):
            u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
        return Z + u.unsqueeze(2) + v.unsqueeze(1)


    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """ Perform Differentiable Optimal Transport in Log-space for stability"""
        
        alpha = self.bin_score
        iters = self.skh_iters
        
        b, m, n = scores.shape
        one = scores.new_tensor(1)
        ms, ns = (m*one).to(scores), (n*one).to(scores)

        if alpha != -1:
            bins0 = alpha.expand(b, m, 1)
            bins1 = alpha.expand(b, 1, n)
            alpha = alpha.expand(b, 1, 1)

            couplings = torch.cat([torch.cat([scores, bins0], -1),
                                torch.cat([bins1, alpha], -1)], 1)
            
            norm = - (ms + ns).log()
            log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
            log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
        else:
            couplings = scores
            
            norm = -ms.log()
            log_mu = (-ms.log()).expand(m)
            log_nu = (-ns.log()).expand(n)
        
       
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

        Z = self.log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
        Z = Z - norm  # multiply probabilities
        return Z
