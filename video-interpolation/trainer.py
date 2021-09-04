import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pytorch_lightning as pl
from torch.functional import Tensor
from model import MLP
import torchmetrics as metrics
from typing import Union, Tuple
import torchvision.io as io
import wandb

from my_utils.resample2d import Resample2d
from my_utils.flow_viz import flow2img


def net(in_channels, activation, out_channels=3):
    return MLP(
        in_channels,
        out_channels=out_channels,
        hidden_dim=256,
        # hidden_dim=512,
        hidden_layers=3,
        # hidden_layers=4,
        activation=activation)


class FlowTrainer(pl.LightningModule):
    def __init__(self, loss=None, lr=None, step=None, flow_scale=None):
        super().__init__()

        self.net = net(3, 'siren', out_channels=2)
        self.resample = Resample2d()
        self.loss = loss
        self.lr = lr
        self.flow_scale = flow_scale

        self.psnr = metrics.PSNR()

    def forward(self, F: Tensor, T: Tensor):
        _, _, h, w = F.shape
        t = T.size(0)
        H = torch.linspace(-1, 1, h).to(T)
        W = torch.linspace(-1, 1, w).to(T)
        gridT, gridH, gridW = torch.meshgrid(T, H, W)
        poses = torch.stack((gridT, gridH, gridW), dim=-1).view(-1, 3)
        flow = self.net(poses).view(t, h, w, 2).permute(0, 3, 1, 2) * self.flow_scale
        return flow

    def on_train_start(self) -> None:
        print(f'=== learning rate {self.lr} ===')

    def training_step(self, batch, batch_idx):
        frame1, frame2, times, gt_flow = batch
        flow = self.forward(frame1, times)
        new_video = self.resample(frame2.contiguous(), flow.contiguous())
        loss = self.loss(new_video, frame1)
        # self.logger.experiment.add_scalar('loss', loss, self.global_step)
        self.log('loss', loss, on_step=True, on_epoch=False)
        epe = torch.sum((flow - gt_flow)**2).sqrt()
        self.log('EPE', epe, on_step=False, on_epoch=True)
        psnr = self.psnr(frame2, new_video)
        self.log('PSNR', psnr, on_step=False, on_epoch=True)
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
