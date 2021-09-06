import torch
import pytorch_lightning as pl
import torchmetrics as metrics
import wandb

from model import MLP
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
    def __init__(self, args, loss=None, flow_scale=None):
        super().__init__()
        self.args = args
        self.net = net(3, 'siren', out_channels=2)
        self.resample = Resample2d()
        self.photo_loss = loss
        self.flow_scale = flow_scale
        self.lr = self.args.lr

        self.psnr = metrics.PSNR()

    def forward(self, F, T):
        _, _, h, w = F.shape
        t = T.size(0)
        H = torch.linspace(-1, 1, h).to(T)
        W = torch.linspace(-1, 1, w).to(T)
        gridT, gridH, gridW = torch.meshgrid(T, H, W)
        poses = torch.stack((gridT, gridH, gridW), dim=-1).view(-1, 3)
        flow = self.net(poses).view(t, h, w, 2).permute(0, 3, 1, 2) * self.flow_scale
        return flow

    def training_step(self, batch, batch_idx):
        frame1, frame2, times, gt_flow = batch
        flow = self.forward(frame1, times)
        new_video = self.resample(frame2.contiguous(), flow.contiguous())

        photo_loss = self.photo_loss(new_video, frame1)
        smooth1_loss = self.smooth_loss(frame1, flow, self.args.edge_func)
        smooth2_loss = self.smooth_loss(frame1, flow, self.args.edge_func, order=2)

        loss = self.args.loss_photo*photo_loss + self.args.loss_smooth1*smooth1_loss + self.args.loss_smooth2*smooth2_loss
        self.log('train/loss', loss.detach(), on_step=True, on_epoch=False)

        epe = torch.sum((flow - gt_flow)**2, dim=1).sqrt().mean().detach()
        psnr = self.psnr(frame2, new_video).detach()
        self.log('metric', {'EPE': epe, 'PSNR': psnr}, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        frame1, _, times, gt_flow = batch
        flow = self.forward(frame1, times)
        flow = torch.cat((flow, gt_flow), dim=-1).permute(0,2,3,1)
        flow = flow.cpu().numpy().clip(-10, 10)
        flow_img = torch.stack([torch.tensor(flow2img(f)) for f in flow]).permute(0,3,1,2)
        return dict(flow=flow_img)

    def validation_epoch_end(self, outputs):
        flow = torch.cat([seq['flow'] for seq in outputs], dim=0).unsqueeze(0)
        if self.logger:
            self.logger.experiment.log({'flow': wandb.Video(flow.type(torch.uint8)),
                                        'epoch': self.current_epoch})

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def smooth_loss(self, img, flow, abs_fun, order=1):
        '''
        First and second order smoothness loss
        Reference: https://github.com/google-research/google-research/
                   blob/235feb2b42f3a56e1e8ed9269186c696c1cecda1/uflow/uflow_utils.py#L743
        '''
        if abs_fun == 'exp':
            abs_fun = torch.abs
        elif abs_fun == 'gauss':
            abs_fun = lambda x: x**2

        img_gx, img_gy = image_grads(img, stride=order)
        flow_gx, flow_gy = image_grads(flow)
        w_x = torch.exp(-abs_fun(self.args.edge_constant * img_gx).mean(dim=1)).unsqueeze(1)
        w_y = torch.exp(-abs_fun(self.args.edge_constant * img_gy).mean(dim=1)).unsqueeze(1)

        if order == 1:
            return ((w_x*robust_l1(flow_gx)).mean() + (w_y*robust_l1(flow_gy)).mean()) / 2
        else:
            flow_gxx, _ = image_grads(flow_gx)
            _, flow_gyy = image_grads(flow_gy)
            return ((w_x*robust_l1(flow_gxx)).mean() + (w_y*robust_l1(flow_gyy)).mean()) / 2
