import torch
import pytorch_lightning as pl
import torchmetrics as metrics
import imageio as io
import wandb

import my_utils.loss as L
import my_utils.occlusions as O
import my_utils.softsplat as S
from my_utils.resample2d import Resample2d
from my_utils.flow_viz import flow2img
from my_utils.utils import *

class FlowTrainer(pl.LightningModule):
    def __init__(self, args, test_tag=None):
        super().__init__()
        self.args = args
        self.net = args.net
        self.resample = Resample2d()
        self.lr = self.args.lr

        self.occlusion = None
        if args.occl == 'brox':
            self.occlusion = O.occlusion_brox
        elif args.occl == 'wang':
            self.occlusion = O.occlusion_wang
        self.l1 = L.L1Loss(args.loss_l1)
        self.census = L.CensusLoss(args.loss_census, max_distance=args.census_width)
        self.ssim = L.SSIMLoss(args.loss_ssim)
        self.smooth1 = L.BilateralSmooth(args.loss_smooth1, args.edge_func, args.edge_constant, 1)

        self.psnr = metrics.PSNR()
        self.test_tag = test_tag
        self.completed_training = False

    def forward(self, F, T, scale):
        _, _, h, w = F.shape
        t = T.size(0)
        H = torch.linspace(-1, 1, h).to(T)
        W = torch.linspace(-1, 1, w).to(T)
        gridT, gridH, gridW = torch.meshgrid(T, H, W)
        poses = torch.stack((gridT, gridH, gridW), dim=-1).view(-1, 3)
        flows = self.net(poses).view(t, h, w, 4).permute(0,3,1,2) * scale
        return flows[:,:2], flows[:,2:]

    def training_step(self, batch, batch_idx):
        frame1, frame2, times, scale = batch[:4]
        flow12, flow21 = self.forward(frame1, times, scale[0])
        if self.occlusion:
            mask1 = self.occlusion(flow12, flow21, self.args.occl_thresh)
            mask2 = self.occlusion(flow21, flow12, self.args.occl_thresh)
        else:
            mask1, mask2 = torch.ones(2)
        if len(batch) == 5:
            with torch.no_grad():
                gt_flow = batch[-1]
                epe = torch.sum((flow12 - gt_flow)**2, dim=1).sqrt().mean().detach()
                self.log('train/EPE', epe, on_step=False, on_epoch=True)

        warped2 = self.resample(frame1, flow21)
        metric = torch.nn.functional.l1_loss(frame2, warped2, reduction='none').mean(1, True)
        softmax1 = S.FunctionSoftsplat(frame2, flow21, -20*metric, strType='softmax')
        # mask1 = mask1 * (softmax1 != 0)
        warped1 = self.resample(frame2, flow12)
        metric = torch.nn.functional.l1_loss(frame1, warped1, reduction='none').mean(1, True)
        softmax2 = S.FunctionSoftsplat(frame1, flow12, -20*metric, strType='softmax')
        # mask2 = mask2 * (softmax2 != 0)

        l1_loss = self.l1(softmax1, frame1, mask1) + self.l1(softmax2, frame2, mask2)
        census_loss = self.census(softmax1, frame1, mask1) + self.census(softmax2, frame2, mask2)
        ssim_loss = self.ssim(softmax1, frame1, mask1) + self.ssim(softmax2, frame2, mask2)
        smooth_loss = self.smooth1(frame1, flow12) + self.smooth1(frame2, flow21)
        loss = l1_loss + census_loss + ssim_loss + smooth_loss

        with torch.no_grad():
            self.log('train/loss', loss, on_step=True, on_epoch=False)
            self.log('train/l1', l1_loss, on_step=True, on_epoch=False)
            self.log('train/census', census_loss, on_step=True, on_epoch=False)
            if ssim_loss != 0:
                self.log('train/ssim', ssim_loss, on_step=True, on_epoch=False)
            self.log('train/smooth', smooth_loss, on_step=True, on_epoch=False)
            psnr = self.psnr(softmax2, frame2)
            self.log('train/PSNR', psnr, on_step=False, on_epoch=True)
        torch.cuda.empty_cache()
        return loss

    def on_train_end(self):
        self.completed_training = True
        return super().on_train_end()

    def validation_step(self, batch, batch_idx):
        frame1, _, times, scale = batch[:4]
        flow_fw, _ = self.forward(frame1, times, scale[0])
        gt_flow = batch[4]
        epe = torch.sum((flow_fw - gt_flow)**2, dim=1).sqrt().mean().detach()
        self.log('val/EPE', epe, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        frame1, _, times, scale = batch[:4]
        flow_fw, flow_bw = self.forward(frame1, times, scale[0])
        flow_img = torch.stack([flow2img(f) for f in flow_fw])
        out = {'flow':flow_img}
        if self.occlusion:
            mask = (self.occlusion(flow_fw, flow_bw, self.args.occl_thresh).type(torch.uint8) * 255).cpu()
            out['mask'] = mask
        if len(batch) == 5:
            gt_flow = batch[-1]
            epe = torch.sum((flow_fw - gt_flow)**2, dim=1).sqrt().mean().detach()
            out['epe'] = epe
        return out

    def test_epoch_end(self, outputs):
        epe = torch.stack([seq['epe'] for seq in outputs]).mean().item() \
              if 'epe' in outputs[0] else 0
        flows = torch.cat([seq['flow'] for seq in outputs], dim=0)
        if self.occlusion:
           masks = torch.cat([seq['mask'] for seq in outputs], dim=0)
        if self.logger:
            self.logger.experiment.log({'flow': wandb.Video(flows.type(torch.uint8)),
                                        'epoch': self.current_epoch})
            if self.occlusion:
                self.logger.experiment.log({'occlusion_mask': wandb.Video(masks),
                                            'epoch': self.current_epoch})
        else:
            flow_gif_file = f'results/flow_{self.test_tag}_epe_{epe:.3f}.gif'
            io.mimsave(flow_gif_file, flows.permute(0,2,3,1), format='GIF', fps=4)
            if self.occlusion:
                mask_gif_file = f'results/occl_{self.test_tag}.gif'
                io.mimsave(mask_gif_file, masks.permute(0,2,3,1), format='GIF', fps=4)
        return epe

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)
