import torch
from torch.nn.functional import interpolate
from torchvision.transforms import Resize, functional
from kornia.filters import BlurPool2D
import pytorch_lightning as pl
import torchmetrics as metrics
import imageio as io
import wandb

import my_utils.loss as L
import my_utils.occlusions as O
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
        if args.loss_photo == 'l1':
            self.photometric = L.L1Loss()
        elif args.loss_photo == 'census':
            self.photometric = L.CensusLoss(max_distance=args.census_width)
        elif args.loss_photo == 'both':
            self.photometric = L.L1CensusLoss(max_distance=args.census_width)
        elif args.loss_photo == 'ssim':
            self.photometric = L.SSIMLoss()
        self.smooth1 = L.BaseLoss() if args.loss_smooth1 == 0 else \
                       L.BilateralSmooth(args.edge_func, args.edge_constant, 1, args.loss_smooth1)
        self.smooth2 = L.BaseLoss() if args.loss_smooth2 == 0 else \
                       L.BilateralSmooth(args.edge_func, args.edge_constant, 2, args.loss_smooth2)
        self.downsample = None
        if args.downsample:
            assert args.downsample_type is not None
            if args.downsample_type == 'nearest':
                self.downsample = Resize(self.args.size // args.downsample)
            elif args.downsample_type == 'bilinear':
                self.downsample = Resize(self.args.size // args.downsample,
                                         interpolation=functional.InterpolationMode.BILINEAR)
            elif args.downsample_type == 'bicubic':
                self.downsample = Resize(self.args.size // args.downsample,
                                         interpolation=functional.InterpolationMode.BICUBIC)
            elif args.downsample_type == 'blurpool':
                self.downsample = BlurPool2D(4, stride=args.downsample)

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
        flow_fw, flow_bw = self.forward(frame1, times, scale[0])
        if self.occlusion:
            mask_fw = self.occlusion(flow_fw, flow_bw, self.args.occl_thresh)
            mask_bw = self.occlusion(flow_bw, flow_fw, self.args.occl_thresh)
        else:
            mask_fw, mask_bw = torch.ones(2)
        if len(batch) == 5:
            gt_flow = batch[-1]
            epe = torch.sum((flow_fw - gt_flow)**2, dim=1).sqrt().mean().detach()
            self.log('train/EPE', epe, on_step=False, on_epoch=True)

        if self.downsample:
            # Enable antialiasing for downsampling the input frame
            new_h = frame1.size(2) // self.args.downsample
            frame1 = Resize(new_h, antialias=True)(frame1)
            frame2 = Resize(new_h, antialias=True)(frame2)
            # No antialising for tensors that require autodiff
            flow_fw = self.downsample(flow_fw) / self.args.downsample
            flow_bw = self.downsample(flow_bw) / self.args.downsample
            if mask_fw.ndim == 4:
                mask_fw = self.downsample(mask_fw)
                mask_bw = self.downsample(mask_bw)

        warped_fw = self.resample(frame2, flow_fw)
        warped_bw = self.resample(frame1, flow_bw)
        photo_loss = self.photometric(warped_fw, frame1, mask_fw) \
                     + self.photometric(warped_bw, frame2, mask_bw)
        smooth1_loss = self.smooth1(frame1, flow_fw) + self.smooth1(frame2, flow_bw)

        loss = photo_loss + smooth1_loss
        self.log('train/loss', loss.detach(), on_step=True, on_epoch=False)

        psnr = self.psnr(warped_fw, frame1).detach()
        self.log('train/PSNR', psnr, on_step=False, on_epoch=True)
        torch.cuda.empty_cache()
        return loss

    def on_train_end(self) -> None:
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
