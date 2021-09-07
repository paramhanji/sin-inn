import torch
from my_utils.resample2d import Resample2d

def image_grads(image_batch, stride=1):
    image_batch_gh = image_batch[:,:,stride:] - image_batch[:,:,:-stride]
    image_batch_gw = image_batch[:,:,:,stride:] - image_batch[:,:,:,:-stride]
    return image_batch_gh, image_batch_gw

def robust_l1(x):
    """Robust L1 metric."""
    return (x**2 + 0.001**2)**0.5

def flow_to_warp(flow):
    """Compute the warp from the flow field"""
    batch, _, ht, wd = flow.shape
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    coords = coords[None].repeat(batch, 1, 1, 1)
    return coords + flow

def occlusion_unity(flow, *args):
    """Placeholder that returns all-True mask"""
    return torch.ones_like(flow[:,0], dtype=torch.bool).unsqueeze(1)

def occlusion_brox(orig_fw, orig_bw):
    """Forward-backward consistency"""
    resample = Resample2d()
    warped_bw = resample(orig_bw.contiguous(), orig_fw.contiguous())
    sq_sum = ((orig_fw + warped_bw)**2).sum(dim=1)
    sum_sq = (orig_fw**2 + warped_bw**2).sum(dim=1)
    return (sq_sum >= 0.01 * sum_sq + 0.5).unsqueeze(1)

def occlusion_wang(bw_flow, threshold=False):
    """Range-map occlusion"""
    # resample = Resample2d()
    # coords = flow_to_warp(bw_flow)
    # coords_floor = torch.floor(coords)
    # coords_offset = coords - coords_floor
    # range_map =
    # if self.threshold:
    # 	return range_map >= 0.75
    # else:
    # 	return 1 = range_map.clamp(0, 1)
    pass
