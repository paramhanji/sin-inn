import torch
import pytorch_lightning as pl
import torchmetrics as metrics
import imageio as io
import wandb

from model import MLP
import my_utils.loss as L
import my_utils.occlusions as O
from my_utils.resample2d import Resample2d
from my_utils.flow_viz import flow2img
from my_utils.utils import *

default_net = MLP(in_channels=3,
                  out_channels=4,
                  hidden_dim=256,
                  hidden_layers=5,
                  activation='siren')

class FlowTrainer(pl.LightningModule):
    def __init__(self, args, net=default_net, test_tag=None):
        super().__init__()
        self.args = args
        self.net = net
        self.resample = Resample2d()
        self.lr = self.args.lr
        if args.occl == 'brox':
            self.occlusion = O.occlusion_brox
        elif args.occl == 'wang':
            self.occlusion = O.occlusion_wang
        else:
            self.occlusion = O.occlusion_unity
        if args.loss_photo == 'l1':
            self.photometric = L.L1Loss()
        elif args.loss_photo == 'census':
            self.photometric = L.CensusLoss()
        elif args.loss_photo == 'ssim':
            self.photometric = L.SSIMLoss()
        self.smooth1 = L.BaseLoss() if args.loss_smooth1 == 0 else \
                       L.BilateralSmooth(args.edge_func, args.edge_constant, 1, args.loss_smooth1)
        self.smooth2 = L.BaseLoss() if args.loss_smooth2 == 0 else \
                       L.BilateralSmooth(args.edge_func, args.edge_constant, 2, args.loss_smooth2)

        self.psnr = metrics.PSNR()
        self.test_tag = test_tag

    def forward(self, F, T, scale):
        _, _, h, w = F.shape
        t = T.size(0)
        H = torch.linspace(-1, 1, h).to(T)
        W = torch.linspace(-1, 1, w).to(T)
        gridT, gridH, gridW = torch.meshgrid(T, H, W)
        poses = torch.stack((gridT, gridH, gridW), dim=-1).view(-1, 3)
        flows = self.net(poses).view(t, h, w, 4).permute(0, 3, 1, 2) * scale
        return flows[:,:2], flows[:,2:]

    def training_step(self, batch, batch_idx):
        frame1, frame2, times, scale = batch[:4]
        flow_fw, flow_bw = self.forward(frame1, times, scale[0])
        warped_fw = self.resample(frame2, flow_fw)
        warped_bw = self.resample(frame1, flow_bw)
        if self.current_epoch > self.args.occl_delay:
            mask_fw = self.occlusion(flow_fw, flow_bw, self.args.occl_thresh)
            mask_bw = self.occlusion(flow_bw, flow_fw, self.args.occl_thresh)
        else:
            mask_fw, mask_bw = torch.ones(2)

        photo_loss = self.photometric(warped_fw, frame1, mask_fw) \
                     + self.photometric(warped_bw, frame2, mask_bw)
        smooth1_loss = self.smooth1(frame1, flow_fw) + self.smooth1(frame2, flow_bw)
        smooth2_loss = self.smooth2(frame1, flow_fw) + self.smooth2(frame2, flow_bw)

        loss = photo_loss + smooth1_loss + smooth2_loss
        self.log('train/loss', loss.detach(), on_step=True, on_epoch=False)

        psnr = self.psnr(warped_fw, frame1).detach()
        self.log('PSNR', psnr, on_step=False, on_epoch=True)
        if len(batch) == 5:
            gt_flow = batch[-1]
            epe = torch.sum((flow_fw - gt_flow)**2, dim=1).sqrt().mean().detach()
            self.log('EPE', epe, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        frame1, _, times, scale = batch[:4]
        flow_fw, flow_bw = self.forward(frame1, times, scale[0])
        mask = (self.occlusion(flow_fw, flow_bw, self.args.occl_thresh).type(torch.uint8) * 255).cpu()
        flow_img = torch.stack([flow2img(f) for f in flow_fw])
        out = {'flow':flow_img, 'mask':mask}
        if len(batch) == 5:
            gt_flow = batch[-1]
            epe = torch.sum((flow_fw - gt_flow)**2, dim=1).sqrt().mean().detach()
            out['epe'] = epe
            self.log('val_epe', epe, on_step=False, on_epoch=True)
        return out

    def validation_epoch_end(self, outputs):
        flows = torch.cat([seq['flow'] for seq in outputs], dim=0).unsqueeze(0)
        masks = torch.cat([seq['mask'] for seq in outputs], dim=0).unsqueeze(0)
        if self.logger:
            self.logger.experiment.log({'flow': wandb.Video(flows.type(torch.uint8)),
                                        'epoch': self.current_epoch})
            self.logger.experiment.log({'occlusion_mask': wandb.Video(masks),
                                        'epoch': self.current_epoch})

    def test_step(self, *args):
        return self.validation_step(*args)

    def test_epoch_end(self, outputs):
        epe = torch.stack([seq['epe'] for seq in outputs]).mean().item() \
              if 'epe' in outputs[0] else 0
        flows = torch.cat([seq['flow'] for seq in outputs], dim=0).permute(0,2,3,1)
        flow_gif_file = f'results/flow_{self.test_tag}_epe_{epe:.3f}.gif'
        io.mimsave(flow_gif_file, flows, format='GIF', fps=4)
        masks = torch.cat([seq['mask'] for seq in outputs], dim=0).permute(0,2,3,1)
        mask_gif_file = f'results/occl_{self.test_tag}.gif'
        io.mimsave(mask_gif_file, masks, format='GIF', fps=4)


    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)
