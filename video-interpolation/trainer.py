import torch
import pytorch_lightning as pl
import torchmetrics as metrics
import wandb

from model import MLP
import loss as L
from my_utils.resample2d import Resample2d
from my_utils.flow_viz import flow2img
from my_utils.utils import *

def net(in_channels, activation, out_channels=3):
    return MLP(
        in_channels,
        out_channels=out_channels,
        hidden_dim=256,
        # hidden_dim=375,
        hidden_layers=3,
        # hidden_layers=5,
        activation=activation)


class FlowTrainer(pl.LightningModule):
    def __init__(self, args, flow_scale=None):
        super().__init__()
        self.args = args
        self.net = net(3, 'siren', out_channels=4)
        self.resample = Resample2d()
        self.flow_scale = flow_scale
        self.lr = self.args.lr
        self.occlusion = occlusion_brox if args.occl == 'brox' else occlusion_unity
        if args.loss_photo == 'l1':
            self.photometric = L.L1Loss(lambda_reg=args.occl_lambda)
        self.smooth1 = L.BaseLoss() if args.loss_smooth1 == 0 else \
                       L.BilateralSmooth(args.edge_func, args.edge_constant, 1, args.loss_smooth1)
        self.smooth2 = L.BaseLoss() if args.loss_smooth2 == 0 else \
                       L.BilateralSmooth(args.edge_func, args.edge_constant, 2, args.loss_smooth2)

        self.psnr = metrics.PSNR()

    def forward(self, F, T):
        _, _, h, w = F.shape
        t = T.size(0)
        H = torch.linspace(-1, 1, h).to(T)
        W = torch.linspace(-1, 1, w).to(T)
        gridT, gridH, gridW = torch.meshgrid(T, H, W)
        poses = torch.stack((gridT, gridH, gridW), dim=-1).view(-1, 3)
        flows = self.net(poses).view(t, h, w, 4).permute(0, 3, 1, 2) * self.flow_scale
        return flows[:,:2], flows[:,2:]

    def training_step(self, batch, batch_idx):
        frame1, frame2, times, gt_flow = batch
        flow_fw, flow_bw = self.forward(frame1, times)
        warped_fw = self.resample(frame2.contiguous(), flow_fw.contiguous())
        warped_bw = self.resample(frame1.contiguous(), flow_bw.contiguous())
        mask_fw = self.occlusion(flow_fw, flow_bw)
        mask_bw = self.occlusion(flow_bw, flow_fw)

        photo_loss = self.photometric(warped_fw, frame1, mask_fw) \
                     + self.photometric(warped_bw, frame2, mask_bw)
        smooth1_loss = self.smooth1(frame1, flow_fw) + self.smooth1(frame2, flow_bw)
        smooth2_loss = self.smooth2(frame1, flow_fw) + self.smooth2(frame2, flow_bw)

        loss = photo_loss + smooth1_loss + smooth2_loss
        self.log('train/loss', loss.detach(), on_step=True, on_epoch=False)

        epe = torch.sum((flow_fw - gt_flow)**2, dim=1).sqrt().mean().detach()
        psnr = self.psnr(warped_fw, frame1).detach()
        self.log('metric', {'EPE': epe, 'PSNR': psnr}, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        frame1, _, times, gt_flow = batch
        flow_fw, flow_bw = self.forward(frame1, times)
        mask = (self.occlusion(flow_fw, flow_bw).type(torch.uint8) * 255).cpu()
        flow_cat = torch.cat((flow_fw, gt_flow), dim=-1).permute(0,2,3,1)
        flow_cat = flow_cat.cpu().numpy().clip(-10, 10)
        flow_img = torch.stack([torch.tensor(flow2img(f)) for f in flow_cat]).permute(0,3,1,2)
        return dict(flow=flow_img, mask=mask)

    def validation_epoch_end(self, outputs):
        flows = torch.cat([seq['flow'] for seq in outputs], dim=0).unsqueeze(0)
        masks = torch.cat([seq['mask'] for seq in outputs], dim=0).unsqueeze(0)
        if self.logger:
            self.logger.experiment.log({'flow': wandb.Video(flows.type(torch.uint8)),
                                        'epoch': self.current_epoch})
            self.logger.experiment.log({'occlusion_mask': wandb.Video(masks),
                                        'epoch': self.current_epoch})

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)
