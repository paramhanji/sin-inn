import torch
from torch.nn import AvgPool2d
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
    def __init__(self, weight=1):
        super().__init__(weight)

    def forward(self, im1, im2, mask):
        return torch.nn.functional.l1_loss(im1*mask, im2*mask) / mask.sum() * mask.numel()


# Photometric losses taken from:
# https://github.com/lliuz/ARFlow/blob/e92a8bbe66f0ced244267f43e3e55ad0fe46ff3e/losses/loss_blocks.py#L7
class CensusLoss(BaseLoss):
    def __init__(self, weight=1, max_distance=2):
        super().__init__(weight=weight)
        self.max_distance = max_distance
        self.patch_size = 2 * max_distance + 1

    def _rgb_to_grayscale(self, image):
        grayscale = image[:, 0, :, :] * 0.2989 + \
                    image[:, 1, :, :] * 0.5870 + \
                    image[:, 2, :, :] * 0.1140
        return grayscale.unsqueeze(1)

    def _ternary_transform(self, image):
        intensities = self._rgb_to_grayscale(image) * 255
        out_channels = self.patch_size * self.patch_size
        w = torch.eye(out_channels).view((out_channels, 1, self.patch_size, self.patch_size))
        weights = w.type_as(image)
        patches = torch.conv2d(intensities, weights, padding=self.max_distance)
        transf = patches - intensities
        transf_norm = transf / torch.sqrt(0.81 + torch.pow(transf, 2))
        return transf_norm

    def _hamming_distance(self, t1, t2):
        dist = torch.pow(t1 - t2, 2)
        dist_norm = dist / (0.1 + dist)
        dist_mean = torch.mean(dist_norm, 1, keepdim=True)  # instead of sum
        return dist_mean

    def _valid_mask(self, t):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * self.max_distance, w - 2 * self.max_distance).type_as(t)
        mask = torch.nn.functional.pad(inner, [self.max_distance] * 4)
        return mask

    def forward(self, im, im_warp, mask):
        t1 = self._ternary_transform(im * mask)
        t2 = self._ternary_transform(im_warp * mask)
        dist = self._hamming_distance(t1, t2)
        valid = self._valid_mask(im)
        return (dist * valid).mean() / mask.sum() * mask.numel()


class L1CensusLoss(CensusLoss):
    def __init__(self, weight=1, max_distance=2):
        super().__init__(weight=weight, max_distance=max_distance)

    def forward(self, im1, im2, mask):
        l1_loss = torch.nn.functional.l1_loss(im1*mask, im2*mask) / mask.sum() * mask.numel()
        census_loss = super().forward(im1, im2, mask)
        return l1_loss + census_loss


class SSIMLoss(BaseLoss):
    def __init__(self, weight=1, md=1):
        super().__init__(weight=weight)
        self.md = md

    def forward(self, x, y, mask):
        x, y = x*mask, y*mask
        patch_size = 2 * self.md + 1
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = AvgPool2d(patch_size, 1, 0)(x)
        mu_y = AvgPool2d(patch_size, 1, 0)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = AvgPool2d(patch_size, 1, 0)(x * x) - mu_x_sq
        sigma_y = AvgPool2d(patch_size, 1, 0)(y * y) - mu_y_sq
        sigma_xy = AvgPool2d(patch_size, 1, 0)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d
        dist = torch.clamp((1 - SSIM) / 2, 0, 1)
        return dist.mean() / mask.sum() * mask.numel()


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
