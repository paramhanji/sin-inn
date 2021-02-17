import torch, numpy as np

def reconstruction(x, y, eps=1e-6):
    # L2 loss
    # return torch.mean((x - y)**2)
    # Charbonnier penalty
    return torch.mean(torch.sqrt((x - y)**2 + eps))

def mmd(x, y, rev=False):
    if rev:
        kernels = [(0.2, 0.1), (0.2, 0.5), (0.2, 2)]
    else:
        kernels = [(0.2, 2), (1.5, 2), (3.0, 2)]

    b, c, h, w = x.shape
    x_flat = x.view(b, c*h*w)
    y_flat = y.view(b, c*h*w)
    xx, yy, xy = torch.mm(x_flat,x_flat.t()), torch.mm(y_flat,y_flat.t()), torch.mm(x_flat,y_flat.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = torch.clamp(rx.t() + rx - 2.*xx, 0, np.inf)
    dyy = torch.clamp(ry.t() + ry - 2.*yy, 0, np.inf)
    dxy = torch.clamp(rx.t() + ry - 2.*xy, 0, np.inf)

    XX, YY, XY = (torch.zeros(xx.shape).to('cuda'),
                  torch.zeros(xx.shape).to('cuda'),
                  torch.zeros(xx.shape).to('cuda'))

    for C,a in kernels:
        XX += C**a * ((C + dxx) / a)**-a
        YY += C**a * ((C + dyy) / a)**-a
        XY += C**a * ((C + dxy) / a)**-a

    return torch.mean(XX + YY - 2.*XY)

def latent_nll(z):
    return torch.mean(z**2)