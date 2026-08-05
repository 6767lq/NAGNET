import functools
from torch.nn import functional as F
import torch

import torch.nn as nn
mse_loss = nn.MSELoss()


def gaussian(window_size, sigma):
    x = torch.arange(window_size).float()
    gauss = torch.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
    return gauss / gauss.sum()
def create_window(window_size, channel):

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

import torch
import torch.nn.functional as F
import cv2
import numpy as np

def gaussian_blur(image, kernel_size=5, sigma=1.0):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def detail_loss(y_true, y_pred):
    # Convert PyTorch tensors to numpy arrays
    y_true_np = y_true.detach().cpu().numpy().transpose(0, 2, 3, 1)
    y_pred_np = y_pred.detach().cpu().numpy().transpose(0, 2, 3, 1)

    detail_true = []
    detail_pred = []

    for i in range(y_true_np.shape[0]):
        # Apply Gaussian blur
        blurred_true = gaussian_blur(y_true_np[i])
        blurred_pred = gaussian_blur(y_pred_np[i])

        # Compute detail layer
        detail_true.append(y_true_np[i] - blurred_true)
        detail_pred.append(y_pred_np[i] - blurred_pred)

    detail_true = np.array(detail_true).transpose(0, 3, 1, 2)
    detail_pred = np.array(detail_pred).transpose(0, 3, 1, 2)

    # Convert back to PyTorch tensors
    detail_true_tensor = torch.tensor(detail_true, dtype=torch.float32).to(y_true.device)
    detail_pred_tensor = torch.tensor(detail_pred, dtype=torch.float32).to(y_pred.device)

    # Compute L2 loss
    loss = F.mse_loss(detail_true_tensor, detail_pred_tensor)
    return loss




def ssim(img1, img2, window_size=11, size_average=True):
    """
    计算SSIM
    """
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim_loss(img1, img2, window_size=11, size_average=True):
    """
    calulate ssim loss
    """
    sssim=ssim(img1, img2, window_size, size_average)

    return 1 - ssim(img1, img2, window_size, size_average)
def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are 'none', 'mean' and 'sum'.

    Returns:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    else:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean'):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights. Default: None.
        reduction (str): Same as built-in losses of PyTorch. Options are
            'none', 'mean' and 'sum'. Default: 'mean'.

    Returns:
        Tensor: Loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if weight is not specified or reduction is sum, just reduce the loss
    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    # if reduction is mean, then compute mean over weight region
    elif reduction == 'mean':
        if weight.size(1) > 1:
            weight = weight.sum()
        else:
            weight = weight.sum() * loss.size(1)
        loss = loss.sum() / weight

    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.5000)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, reduction='sum')
    tensor(3.)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', **kwargs):


        #
        pre_gt=pred[0]
        #
        pre_noise=pred[1]
        target_gt=target[0]
        target_noise=target[1]
        lossf=ssim_loss(pre_noise,target_noise)+mse_loss(pre_gt,target_gt)



        return lossf

    return wrapper
