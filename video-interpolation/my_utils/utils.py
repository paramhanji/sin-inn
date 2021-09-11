import torch

def image_grads(image_batch, stride=1):
    image_batch_gh = image_batch[:,:,stride:] - image_batch[:,:,:-stride]
    image_batch_gw = image_batch[:,:,:,stride:] - image_batch[:,:,:,:-stride]
    return image_batch_gh, image_batch_gw

def robust_l1(x):
    """Robust L1 metric."""
    return (x**2 + 0.001**2)**0.5

def abs_robust_loss(diff, eps=0.01, q=0.4):
  """The so-called robust loss used by DDFlow."""
  return torch.pow((torch.abs(diff) + eps), q)

def flow_to_warp(flow):
    """Compute the warp from the flow field"""
    batch, _, ht, wd = flow.shape
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    coords = coords[None].repeat(batch, 1, 1, 1)
    return coords + flow
