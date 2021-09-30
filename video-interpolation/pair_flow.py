#%%
import torch, tqdm
import matplotlib.pyplot as plt
import model as M
import progressive_controller as C
import data as D
from my_utils.resample2d import Resample2d
from my_utils.flow_viz import flow2img
import my_utils.loss as L
import my_utils.softsplat as S
import my_utils.occlusions as O
import warnings

def plot_img_grid(*imgs):
    # print()
    # return
    N = len(imgs)
    _, axes = plt.subplots(N, 1, squeeze=False, dpi=300)
    for ax, im in zip(axes[:,0], imgs):
        if torch.is_tensor(im) and im.size(1) == 2:
            im = flow2img(im[0].cpu(), clip=10)
        if torch.is_tensor(im):
            im = im.squeeze().permute(1,2,0).cpu()
        ax.imshow(im)
        ax.axis('off')
    plt.show()

# %%
video = D.Images('../datasets/sintel/training/final/bamboo_2', 436)
frame1, frame2, _, scale, gt = video[40]
_, h, w = frame1.shape
plot_img_grid(frame1, frame2, gt[None])
frame1, frame2, gt = frame1[None].cuda(), frame2[None].cuda(), gt.cuda()

# %%
print('prbf12_mask')
epochs = 2500
params = M.ModelParams(domain_dim=2, std_rbf=12)
net = M.model_dict['PRBF'](params).cuda()
if net.is_progressive:
    net = C.LinearControllerEarly(net, epochs, epsilon=1e-3)
resample = Resample2d()
l1 = L.L1Loss(1)
census = L.CensusLoss(0.1, max_distance=3)
smooth1 = L.BilateralSmooth(0.1, 'gauss', 150, 1)
occlusion = O.occlusion_wang
opt = torch.optim.Adam(net.parameters(), lr=5e-4)

# %%
pbar = tqdm.trange(epochs)
for epoch in pbar:
    opt.zero_grad()
    H = torch.linspace(-1, 1, h)
    W = torch.linspace(-1, 1, w)
    gridH, gridW = torch.meshgrid(H, W)
    poses = torch.stack((gridH, gridW), dim=-1).view(-1, 2).cuda()
    flows = net(poses).view(h, w, 4).permute(2,0,1) * scale
    flow12, flow21 = flows[None,:2], flows[None,2:]

    mask1 = occlusion(flow12, flow21, 0.7)
    mask2 = occlusion(flow21, flow12, 0.7)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        warped2 = resample(frame1, flow21)
        metric = torch.nn.functional.l1_loss(frame2, warped2, reduction='none').mean(1, True)
        softmax1 = S.FunctionSoftsplat(frame2, flow21, -20*metric, strType='softmax')
        mask1 = mask1 * (softmax1 != 0)
        warped1 = resample(frame2, flow12)
        metric = torch.nn.functional.l1_loss(frame1, warped1, reduction='none').mean(1, True)
        softmax2 = S.FunctionSoftsplat(frame1, flow12, -20*metric, strType='softmax')
        mask2 = mask2 * (softmax2 != 0)

    l1_loss = l1(softmax1, frame1, mask1) + l1(softmax2, frame2, mask2)
    census_loss = census(softmax1, frame1, mask1) + census(softmax2, frame2, mask2)
    smooth_loss = smooth1(frame1, flow12) + smooth1(frame2, flow21)
    loss = l1_loss + census_loss + smooth_loss
    # loss = ((gt - flow12)**2).mean()
    loss.backward()
    opt.step()
    net.stash_iteration(loss)

    with torch.no_grad():
        epe = ((flow12 - gt)**2).sum(dim=0).sqrt().mean().item()
        pbar.set_postfix({'l1':l1_loss.item(), 'census':census_loss.item(), 'smooth':smooth_loss.item(), 'epe':epe})
        torch.cuda.empty_cache()
        if (epoch + 1) % (epochs//5) == 0:
            print()
            # plot_img_grid(flow12)

plot_img_grid(flow12.detach())


# %%
