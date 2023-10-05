import torch
import torch.nn as nn
from copy import deepcopy
from ..registry import COMPONENTS
from ..builder import build_backbone


EPS = 1e-8


def normalize_fn(mat):
    return (mat - mat.min()) / (mat.max() - mat.min() + EPS)

@COMPONENTS.register_module()
class Regularizer:
    def update(self):
        """ Stub method """
        raise NotImplementedError

    def penalty(self):
        """ Stub method """
        raise NotImplementedError

    def state_dict(self):
        """ Stub method """
        raise NotImplementedError

    def load_state_dict(self, state):
        """ Stub method """
        raise NotImplementedError
    
    
    
@COMPONENTS.register_module()
class EWC_(Regularizer):
    # note: by taking in consideration the torch.distributed package and that the update is only computed by the rank 0,
    #       we can save memory in other ranks. Actually it's not useful because I use GPU with the same memory.
    def __init__(self, model, model_old, fisher=None, alpha=0.9, normalize=True):

        self.model = build_backbone(model)
        self.alpha = alpha
        self.normalize = normalize

        # store old model for penalty step
        if model_old is not None:
            self.model_old = build_backbone(model_old)
            self.model_old_dict = self.model_old.state_dict()
            self.penalize = True
        else:
            self.penalize = False

        # make the fisher matrix for the estimate of parameter importance
        # store the old fisher matrix (if exist) for penalize step
        if fisher is not None:  # initialize the old Fisher Matrix
            self.fisher_old = fisher
            self.fisher = {}
            for key, par in self.fisher_old.items():
                self.fisher_old[key].requires_grad = False
                self.fisher_old[key] = normalize_fn(par) if normalize else par
                self.fisher_old[key] = self.fisher_old[key].cuda()
                self.fisher[key] = torch.clone(par).cuda()
        else:  # initialize a new Fisher Matrix and don't penalize, we miss an information
            self.fisher_old = None
            self.penalize = False
            self.fisher = {}

        self.fisher = {}
        for n, p in self.model.named_parameters():  # update fisher with new keys (due to incremental classes)
            if p.requires_grad and n not in self.fisher:
                self.fisher[n] = torch.ones_like(p, requires_grad=False).cuda()

    def update(self, model):
        # suppose model have already grad computed, so we can directly update the fisher by getting model.parameters
        for n, p in model.named_parameters():
            self.fisher[n] = (self.alpha * (p.grad ** 2)) + ((1 - self.alpha) * self.fisher[n])

    def penalty(self, model):
        if not self.penalize:
            return 0.
        else:
            loss = 0.
            for n, p in model.named_parameters():
                if n in self.model_old_dict and p.requires_grad:
                    loss += (self.fisher_old[n] * (p - self.model_old_dict[n]) ** 2).sum()
            return loss

    def get(self):
        return self.fisher  # return the new Fisher matrix


# @COMPONENTS.register_module()
# class EWC(object):
    
#     """
#     Class to calculate the Fisher Information Matrix
#     used in the Elastic Weight Consolidation portion
#     of the loss function
#     """
    
#     def __init__(self, model: nn.Module, dataset: list):

#         self.model = model #pretrained model
#         self.dataset = dataset #samples from the old task or tasks
        
#         # n is the string name of the parameter matrix p, aka theta, aka weights
#         # in self.params we reference all of those weights that are open to
#         # being updated by the gradient
#         self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        
#         # make a copy of the old weights, ie theta_A,star, ie 𝜃∗A, in the loss equation
#         # we need this to calculate (𝜃 - 𝜃∗A)^2 because self.params will be changing 
#         # upon every backward pass and parameter update by the optimizer
#         self._means = {}
#         for n, p in deepcopy(self.params).items():
#             self._means[n] = var2device(p.data)
        
#         # calculate the fisher information matrix 
#         self._precision_matrices = self._diag_fisher()

#     def _diag_fisher(self):
        
#         # save a copy of the zero'd out version of
#         # each layer's parameters of the same shape
#         # to precision_matrices[n]
#         precision_matrices = {}
#         for n, p in deepcopy(self.params).items():
#             p.data.zero_()
#             precision_matrices[n] = var2device(p.data)

#         # we need the model to calculate the gradient but
#         # we have no intention in this step to actually update the model
#         # that will have to wait for the combining of this EWC loss term
#         # with the new task's loss term
#         self.model.eval()
#         for input in self.dataset:
#             self.model.zero_grad()
#             # remove channel dim, these are greyscale, not color rgb images
#             # bs,1,h,w -> bs,h,w
#             input = input.squeeze(1)
#             input = var2device(input)
#             output = self.model(input).view(1, -1)
#             label = output.max(1)[1].view(-1)
#             # calculate loss and backprop
#             loss = F.nll_loss(F.log_softmax(output, dim=1), label)
#             loss.backward()

#             for n, p in self.model.named_parameters():
#                 precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

#         precision_matrices = {n: p for n, p in precision_matrices.items()}
#         return precision_matrices

#     def penalty(self, model: nn.Module):
#         loss = 0
#         for n, p in model.named_parameters():
#             _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
#             loss += _loss.sum()
#         return loss 