# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpt.models.builder import build_loss
from mmpt.datasets.flyingthingsplus.utils import *

from mmpt.utils.util import tensor2img

from ..registry import LOSSES
from .utils import *

@masked_loss
def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated L1 loss.
    """
    return F.l1_loss(pred, target, reduction='none')

@masked_loss
def smooth_l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated L1 loss.
    """
    return F.smooth_l1_loss(pred, target, reduction='none')

@masked_loss
def mse_loss(pred, target):
    """MSE loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated MSE loss.
    """
    return F.mse_loss(pred, target, reduction='none')

@masked_loss
def charbonnier_loss(pred, target, eps=1e-12):
    """Charbonnier loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated Charbonnier loss.
    """
    return torch.sqrt((pred - target)**2 + eps)

@masked_loss
def kl_loss(pred, target, eps=1e-12):
    """Charbonnier loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated Charbonnier loss.
    """
    return F.kl_div(F.log_softmax(pred, dim=-1), target.softmax(dim=-1), reduction='none')

@LOSSES.register_module()
class Ce_Loss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * F.cross_entropy(pred, target.squeeze(1), reduction=self.reduction)

@LOSSES.register_module()
class Soft_Ce_Loss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
  
        log_likelihood = -F.log_softmax(pred, dim=-1)
        bsz = pred.shape[0]
        if self.reduction == 'mean':
            loss = torch.sum(torch.mul(log_likelihood, target.softmax(dim=-1))) / bsz
        else:
            loss = torch.sum(torch.mul(log_likelihood, target.softmax(dim=-1)), dim=-1)

        if weight is not None:
            weight = weight.reshape(-1)
            loss = (loss * weight).sum() / (weight.sum()+1e-7)

        return loss * self.loss_weight

@LOSSES.register_module()
class Balance_Ce_Loss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        
    def forward(self, pred, gt, valid):
        # pred and gt are the same shape
        for (a,b) in zip(pred.size(), gt.size()):
            assert(a==b) # some shape mismatch!
            
        if valid is not None:
            for (a,b) in zip(pred.size(), valid.size()):
                assert(a==b) # some shape mismatch!
        else:
             valid = torch.ones_like(gt)
            
        pos = (gt > 0.95).float()
        neg = (gt < 0.05).float()

        label = pos*2.0 - 1.0
        a = -label * pred
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))
        
        pos_loss = basic.reduce_masked_mean(loss, pos)
        neg_loss = basic.reduce_masked_mean(loss, neg)

        balanced_loss = pos_loss + neg_loss

        return balanced_loss * self.loss_weight





@LOSSES.register_module()
class Kl_Loss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * kl_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


@LOSSES.register_module()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduce loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


@LOSSES.register_module()
class SmoothL1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduce loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


@LOSSES.register_module()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        # if self.sample_wise:
        #     loss = F.mse_loss(pred, target, reduction='none')
        #     if self.reduction == 'mean':
        #         loss = loss.mean(-1)
        #     else:
        #         loss = loss.sum(-1)
        # else:
        #     assert self.reduction != 'none'
        #     loss = F.mse_loss(pred, target, reduction=self.reduction)
        # return self.loss_weight * loss
        return self.loss_weight * mse_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)



@LOSSES.register_module()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean',
                 sample_wise=False,
                 eps=1e-12):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


@LOSSES.register_module()
class CosineSimLoss(nn.Module):
    """NLL Loss.

    It will calculate Cosine Similarity loss given cls_score and label.
    """

    def __init__(self,
                 loss_weight=1.0,
                 with_norm=True,
                 negative=False,
                 pairwise=False,
                 reduction='mean',
                 **kwargs):
        super().__init__()
        self.with_norm = with_norm
        self.negative = negative
        self.pairwise = pairwise
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, cls_score, label, mask=None, **kwargs):
        if self.with_norm:
            cls_score = F.normalize(cls_score, p=2, dim=1)
            label = F.normalize(label, p=2, dim=1)
        if mask is not None:
            assert self.pairwise
        if self.pairwise:
            cls_score = cls_score.flatten(2)
            label = label.flatten(2)
            prod = torch.einsum('bci,bcj->bij', cls_score, label)
            if mask is not None:
                assert prod.shape == mask.shape
                prod *= mask.float()
            prod = prod.flatten(1)
        else:
            prod = torch.sum(
                cls_score * label, dim=1).view(cls_score.size(0), -1)
        if self.negative:
            loss = -prod.mean(dim=-1)
        else:
            loss = 2 - 2 * prod.mean(dim=-1)
        
        # if self.reduction == 'mean':
        #     loss = loss.mean()
        # elif self.reduction == 'sum':
        #     loss = loss.sum()
        # else:
        #     pass
            
        return loss * self.loss_weight
    
    

@LOSSES.register_module()
class DiscreteLoss(nn.Module):
    """NLL Loss.

    It will calculate Cosine Similarity loss given cls_score and label.
    """

    def __init__(self,
                 nbins,
                 fmax,
                 loss_weight=1.0,
                 reduction='mean',
                 **kwargs):
        super().__init__()
      
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss = nn.CrossEntropyLoss()
        assert nbins % 2 == 1, "nbins should be odd"
        self.nbins = nbins
        self.fmax = fmax
        self.step = 2 * fmax / float(nbins)
        
    def tobin(self, target):
        target = torch.clamp(target, -self.fmax + 1e-3, self.fmax - 1e-3)
        quantized_target = torch.floor((target + self.fmax) / self.step)
        return quantized_target.type(torch.cuda.LongTensor)

    def __call__(self, input, target):
        size = target.shape[2:4]
        if input.shape[2] != size[0] or input.shape[3] != size[1]:
            input = nn.functional.interpolate(input, size=size, mode="bilinear", align_corners=True)
        target = self.tobin(target)
        assert input.size(1) == self.nbins * 2
        loss = self.loss(input[:,:self.nbins,...], target[:,0,...]) + self.loss(input[:,self.nbins:,...], target[:,1,...])
        return loss * self.loss_weight




@LOSSES.register_module()
class Kl_Loss_Gaussion(nn.Module):
    """kl div betweeen two gaussion (VAE)

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        mu_pred, var_pred = pred
        mu_target, var_target = target
        return self.kl_criterion(mu_pred, var_pred, mu_target, var_target, weight)


    def kl_criterion(self, mu1, logvar1, mu2, logvar2, weight=None):
        bsz = mu1.shape[0]
        sigma1 = logvar1.mul(0.5).exp() 
        sigma2 = logvar2.mul(0.5).exp() 

        kld = torch.log(sigma2/(sigma1+1e-7)) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2

        if weight is not None:
            return  (kld * weight).sum() / weight.sum()
        else:
            return kld.mean()


@LOSSES.register_module()
class Kl_Loss_Laplace(nn.Module):
    """kl div betweeen two Laplace (VAE)

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        mu_pred, var_pred = pred
        mu_target, var_target = target
        return self.kl_criterion(mu_pred, var_pred, mu_target, var_target, weight)


    def kl_criterion(self, mu1, logvar1, mu2, logvar2, weight=None):
        bsz = mu1.shape[0]

        term1 = ( logvar1 * torch.exp(-(torch.abs(mu1-mu2))/logvar1) + torch.abs(mu1-mu2) ) / logvar2

        term2 = torch.log(logvar2/(logvar1+1e-9)) - 1

        kld = term1 + term2

        if weight is not None:
            return  (kld * weight).sum() / weight.sum()
        else:
            return kld.mean()
    
@LOSSES.register_module()
class SmoothnessLoss(nn.Module):
    """spatial smoothness loss

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False, edge_weighted=True, order='first',edge_weighting_fn='exp'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

        self.edge_weighted = edge_weighted
        self.order = order
        self.edge_weighting_fn = edge_weighting_fn
    
    @staticmethod
    def weight_fn(x, mode='exp', constant=150.0):
        # B x H x W x 3 -> x
        if mode == 'exp':
            x = torch.exp(-torch.abs(constant * x).mean(-1, keepdim=True))
            return x.permute(0,3,1,2)
        else:
            raise NotImplementedError

    @staticmethod
    def compute_grads(image_batch, stride=1):
        image_batch_gh = image_batch[:, stride:] - image_batch[:, :-stride]
        image_batch_gw = image_batch[:, :, stride:] - image_batch[:, :, :-stride]
        return image_batch_gh, image_batch_gw

    
    def second_order_smoothness_loss(self, image, flow):
        """Computes a second-order smoothness loss.
        Computes a second-order smoothness loss (only considering the non-mixed
        partial derivatives).
        Args:
            image: Image used for the edge-aware weighting [batch, height, width, 2].
            flow: Flow field for with to compute the smoothness loss [batch, height,
            width, 2].
            edge_weighting_fn: Function used for the edge-aware weighting.
        Returns:
            Average second-order smoothness loss.
        """
        b, h, w, c = flow.shape

        # B x (H-2) x (W-2) x 3
        img_gx, img_gy = self.compute_grads(image, stride=2)

        # B x (H-2) x (W-2) x 1
        weights_xx = self.weight_fn(img_gx)
        weights_yy = self.weight_fn(img_gy)

        weights_xx = F.interpolate(weights_xx, size=(h-2,w)).permute(0,2,3,1)
        weights_yy = F.interpolate(weights_yy, size=(h,w-2)).permute(0,2,3,1)

        # Compute second derivatives of the predicted smoothness.
        # B x (H-1) x (W-1) x C
        flow_gx, flow_gy = self.compute_grads(flow)
        # B x (H-2) x (W-2) x 3
        flow_gxx, _ = self.compute_grads(flow_gx)
        _, flow_gyy = self.compute_grads(flow_gy)

         # Compute weighted smoothness
        smooth_x_loss = l1_loss(
            flow_gxx,
            torch.zeros_like(flow_gxx).cuda(),
            weights_xx.repeat(1,1,1,flow_gx.shape[-1]),
            reduction=self.reduction,
            sample_wise=self.sample_wise)

        smooth_y_loss = l1_loss(
            flow_gyy,
            torch.zeros_like(flow_gyy).cuda(),
            weights_yy.repeat(1,1,1,flow_gyy.shape[-1]),
            reduction=self.reduction,
            sample_wise=self.sample_wise)

        return smooth_x_loss + smooth_y_loss / 2


    def first_order_smoothness_loss(self, image, flow):
        """Computes a first-order smoothness loss.
        Args:
            image: Image used for the edge-aware weighting [batch, height, width, 2].
            flow: Flow field for with to compute the smoothness loss [batch, height,
            width, 2].
            edge_weighting_fn: Function used for the edge-aware weighting.
        Returns:
            Average first-order smoothness loss.
        """
        b, h, w, c = flow.shape

        # B x (H-1) x (W-1) x 3
        img_gx, img_gy = self.compute_grads(image)
        # B x (H-1) x (W-1) x 1
        weights_x = self.weight_fn(img_gx)
        weights_y = self.weight_fn(img_gy)

        weights_x = F.interpolate(weights_x, size=(h-1,w)).permute(0,2,3,1)
        weights_y = F.interpolate(weights_y, size=(h,w-1)).permute(0,2,3,1)

        # Compute second derivatives of the predicted smoothness.
        # B x (H-1) x (W-1) x C 
        flow_gx, flow_gy = self.compute_grads(flow)

        # Compute weighted smoothness
        smooth_x_loss = l1_loss(
            flow_gx,
            torch.zeros_like(flow_gx).cuda(),
            weights_x.repeat(1,1,1,flow_gx.shape[-1]),
            reduction=self.reduction,
            sample_wise=self.sample_wise)

        smooth_y_loss = l1_loss(
            flow_gy,
            torch.zeros_like(flow_gy).cuda(),
            weights_y.repeat(1,1,1,flow_gy.shape[-1]),
            reduction=self.reduction,
            sample_wise=self.sample_wise)

        return smooth_x_loss + smooth_y_loss / 2

    def forward(self, pred, image=None, **kwargs):
        if self.edge_weighted: assert image is not None
        if self.order == 'first':
            loss = self.first_order_smoothness_loss(image, pred)
        else:
            loss = self.second_order_smoothness_loss(image, pred)
            
        return loss