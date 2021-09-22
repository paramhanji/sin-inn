import torch, numpy as np

TAG_CHAR = np.array([202021.25], np.float32)


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

def readFlow(fn):
    """
    Read .flo file in Middlebury format
    Reference: http://stackoverflow.com/questions/
               28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    WARNING: this will work on little-endian architectures (eg Intel x86) only!
    """
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()
