def image_grads(image_batch, stride=1):
    image_batch_gh = image_batch[:,:,stride:] - image_batch[:,:,:-stride]
    image_batch_gw = image_batch[:,:,:,stride:] - image_batch[:,:,:,:-stride]
    return image_batch_gh, image_batch_gw

def robust_l1(x):
    """Robust L1 metric."""
    return (x**2 + 0.001**2)**0.5
