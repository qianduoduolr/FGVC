import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


def ce_loss(pred, target, reduction='mean'):
    return F.cross_entropy(pred, target, reduction=reduction)


@LOSSES.register_module()
class Nce_Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        """NCE Loss only has one positive

        Args:
            loss_weight (float, optional): [description]. Defaults to 1.0.
            reduction (str, optional): [description]. Defaults to 'mean'.
            sample_wise (bool, optional): [description]. Defaults to False.

        Raises:
            ValueError: [description]
        """
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, M). Predicted tensor.
            weight (Tensor, optional): of shape (N, M). Element-wise
                weights. Default: None.
        """
        label = torch.zeros([pred.shape[0]]).long().to(pred.device)
        return self.loss_weight * ce_loss(pred, label, self.reduction)

@LOSSES.register_module()
class Multi_Nce_Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False, mode='in_sum'):
        """NCE Loss only has multiply positive

        Args:
            loss_weight (float, optional): [description]. Defaults to 1.0.
            reduction (str, optional): [description]. Defaults to 'mean'.
            sample_wise (bool, optional): [description]. Defaults to False.

        Raises:
            ValueError: [description]
        """
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.mode = mode

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, M). Predicted tensor.
            target (Tensor): of shape (N, M). Ground truth tensor.
            weight (Tensor, optional): of shape (N, M). Element-wise
                weights. Default: None.
        """
        if self.mode == 'in_sum':
            loss = - torch.log( (F.softmax(pred, dim=1) * target ).sum(1))
        else:
            loss = - torch.log( (F.softmax(pred, dim=1)) * target ) / target.sum(1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise NotImplementedError