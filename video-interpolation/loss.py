import torch
from my_utils.resample2d import Resample2d
from my_utils.utils import *


class BaseLoss(torch.nn.Module):
    """Placeholder that returns 0 and also serves as a base class"""
    def __init__(self, weight=0):
        super().__init__()
        self.weight = weight

    def forward(self, *args):
        return 0

class L1Loss(BaseLoss):
    """Wrapper for torch.nn.l1 to use occlusion mask"""
    def __init__(self, lambda_reg, weight=1):
        super().__init__(weight)
        self.lambda_reg = lambda_reg

    def forward(self, im1, im2, mask):
        if mask.dtype == torch.bool:
            return torch.nn.functional.l1_loss(torch.masked_select(im1, mask),
                                               torch.masked_select(im2, mask)) * self.weight + \
                   torch.logical_not(mask).sum() / mask.numel() * self.lambda_reg
        else:
            # Soft mask requires matrix multiplication with separate num and denom
            raise NotImplementedError


class BilateralSmooth(BaseLoss):
    """Edge-aware First and second order smoothness loss"""
    def __init__(self, abs_fun, edge_constant, order, weight):
        super().__init__(weight)
        if abs_fun == 'exp':
            self.abs_fun = torch.abs
        elif abs_fun == 'gauss':
            self.abs_fun = lambda x: x**2
        self.edge_constant = edge_constant
        self.order = order

    def forward(self, img, flow):
        img_gx, img_gy = image_grads(img, stride=self.order)
        flow_gx, flow_gy = image_grads(flow)
        w_x = torch.exp(-self.abs_fun(self.edge_constant * img_gx).mean(dim=1)).unsqueeze(1)
        w_y = torch.exp(-self.abs_fun(self.edge_constant * img_gy).mean(dim=1)).unsqueeze(1)

        if self.order == 1:
            loss = ((w_x*robust_l1(flow_gx)).mean() + (w_y*robust_l1(flow_gy)).mean()) / 2
        elif self.order == 2:
            flow_gxx, _ = image_grads(flow_gx)
            _, flow_gyy = image_grads(flow_gy)
            loss = ((w_x*robust_l1(flow_gxx)).mean() + (w_y*robust_l1(flow_gyy)).mean()) / 2
        return loss * self.weight
