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

def plot_img_grid(*imgs):
    # print()
    # return
    N = len(imgs)
    _, axes = plt.subplots(N, 1, squeeze=False, dpi=300)
    for ax, im in zip(axes[:,0], imgs):
        if torch.is_tensor(im) and im.size(0) == 2:
            im = flow2img(im.cpu(), clip=10)
        if torch.is_tensor(im):
            im = im.permute(1,2,0).cpu()
        ax.imshow(im)
        ax.axis('off')
    plt.show()

# %%
video = D.Images('../datasets/sintel/training/final/ambush_2', 436)
fr1, fr2, _, scale, gt = video[1]
_, h, w = fr1.shape
plot_img_grid(fr1, fr2, gt)
fr1, fr2, gt = fr1.cuda(), fr2.cuda(), gt.cuda()

epochs = 5000
params = M.ModelParams(domain_dim=2)
net = M.model_dict['RBF'](params).cuda()
if net.is_progressive:
    net = C.LinearControllerEarly(net, epochs, epsilon=1e-3)
resample_fn = Resample2d()
loss_photo = L.L1CensusLoss(max_distance=3)
loss_smooth = L.BilateralSmooth('gauss', 150, 1, .1)
occlusion = O.occlusion_wang

opt = torch.optim.Adam(net.parameters(), lr=1e-4)

# %%
pbar = tqdm.trange(epochs)
for epoch in pbar:
    opt.zero_grad()
    H = torch.linspace(-1, 1, h)
    W = torch.linspace(-1, 1, w)
    gridH, gridW = torch.meshgrid(H, W)
    poses = torch.stack((gridH, gridW), dim=-1).view(-1, 2).cuda()
    flows = net(poses).view(h, w, 4).permute(2,0,1) * scale
    warped_f = resample_fn(fr2[None], flows[None,:2])
    warped_b = resample_fn(fr1[None], flows[None,2:])
    # mask = torch.ones(1).cuda()
    mask_f = occlusion(flows[None,:2], flows[None,2:], 0.7)
    mask_b = occlusion(flows[None,2:], flows[None,:2], 0.7)
    smooth = loss_smooth(fr1[None], flows[None,:2]) + loss_smooth(fr2[None], flows[None,2:])

    metric = torch.nn.functional.l1_loss(fr1[None], warped_f, reduction='none').mean(1, True)
    softmax_f = S.FunctionSoftsplat(fr1[None], flows[None,:2], -20*metric, strType='softmax')
    metric = torch.nn.functional.l1_loss(fr2[None], warped_b, reduction='none').mean(1, True)
    softmax_b = S.FunctionSoftsplat(fr2[None], flows[None,2:], -20*metric, strType='softmax')
    photo = loss_photo(softmax_f, fr2[None], mask_b) + loss_photo(softmax_b, fr1[None], mask_f)
    # photo = loss_photo(warped, fr1[None], mask)
    loss = photo + smooth
    # loss = ((gt - flows[:2])**2).mean()
    loss.backward()
    opt.step()

    with torch.no_grad():
        epe = ((flows[:2] - gt)**2).sum(dim=0).sqrt().mean().item()
        pbar.set_postfix({'photo':photo.item(), 'smooth':smooth.item(), 'epe':epe})
        torch.cuda.empty_cache()
        if (epoch + 1) % (epochs//10) == 0:
            plot_img_grid(flows[:2])


# %%
