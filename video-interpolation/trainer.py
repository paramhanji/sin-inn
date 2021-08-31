import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pytorch_lightning as pl
from torch.functional import Tensor
from model import MLP
import torchmetrics as metrics
from typing import Union, Tuple

from resample2d import Resample2d


class FlowTrainer(pl.LightningModule):
    def __init__(self, *,
            encoder=nn.Identity(), 
            net=MLP(2, 3, 256, 3, activation='relu'),
            loss=F.mse_loss,
            lr=1e-3):
        super().__init__()

        self.net = net
        self.encoder = encoder
        self.resample = Resample2d()
        self.loss = loss
        self.lr = lr
        self.data_step = 10
        self.write_gt_flow = True

        self.psnr = metrics.PSNR()

    def forward(self, F: Tensor, T: Tensor, factor=1, gt_flow=None):
        if gt_flow is not None:
            flow = gt_flow * factor
        else:
            h, w = F.shape[-2:]
            t = T.size(0)
            H = torch.linspace(-1, 1, h).to(T)
            W = torch.linspace(-1, 1, w).to(T)
            gridT, gridH, gridW = torch.meshgrid(T, H, W)
            poses = torch.stack((gridT, gridH, gridW), dim=-1).view(-1, 3)
            encoding = self.encoder(poses)
            flow = self.net(encoding).view(t, h, w, 2).permute(0, 3, 1, 2) * factor * self.flow_scale
        pred = self.resample(F.contiguous(), flow.contiguous())
        return pred, flow

    def on_train_start(self) -> None:
        self.flow_scale = self.trainer.train_dataloader.dataset.datasets.flow.max()
        print(f'=== learning rate {self.lr} ===')

    def training_step_end(self, outputs):
        self.logger.experiment.add_scalar('Loss', outputs['loss'], self.global_step)
        self.logger.experiment.add_scalar('PSNR', self.psnr(outputs['target'],outputs['predict']), self.global_step)
        return outputs['loss']

    def training_epoch_end(self, outputs):
        dataset = self.trainer.train_dataloader.dataset.datasets
        self.data_step = dataset.step
        frames = dataset.videos
        if self.current_epoch == 0:
            self.logger.experiment.add_video('original', frames.unsqueeze(0), 0)

    def training_step(self, batch, batch_idx):
        frame1, frame2, times, flow = batch
        new_video, new_flow = self.forward(frame1, times)
        loss = self.loss(new_video, frame2)
        # mask = flow != 0
        # loss = ((new_flow[mask] - flow[mask])**2).mean()
        self.log('train_loss', loss)
        return dict(loss=loss, predict=new_video.clamp(0, 1), target=frame2)

    def validation_step(self, batch, batch_idx):
        frame1, frame2, times, gt_flow = batch
        vid_all = [frame1]
        flow_all = []
        for i, t in enumerate(torch.linspace(0, 1, self.data_step + 1)[1:-1]):
            if self.write_gt_flow:
                new_video, flow = self.forward(frame1, times, factor=t, gt_flow=gt_flow)
            else:
                new_video, flow = self.forward(frame1, times, factor=t)
            flow = flow.norm(dim=1)
            vid_all.append(new_video)
            if i == 0:
                flow_all.append(flow/t)
        vid_all.append(frame2)
        vid_all = torch.stack(vid_all).squeeze()
        flow_all = torch.stack(flow_all).squeeze(1)
        flow_all = torch.cat((flow_all, gt_flow.norm(dim=1)), dim=-1)
        return dict(vid=vid_all, flow=flow_all)

    def validation_epoch_end(self, outputs):
        video = torch.cat([seq['vid'] for seq in outputs]).unsqueeze(0)
        flow = torch.stack([seq['flow'] for seq in outputs]).unsqueeze(0)
        self.logger.experiment.add_video('val_video', video, self.current_epoch)
        self.logger.experiment.add_video('val_flow', flow, self.current_epoch)
        if self.write_gt_flow:
            print('Logging GT flow')
            self.write_gt_flow = False

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)
