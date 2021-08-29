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

class VideoTrainer(pl.LightningModule):
    def __init__(self, *,
            encoder=nn.Identity(), 
            mask=None,
            net=MLP(2, 3, 256, 3, activation='relu'),
            loss=F.mse_loss,
            lr=1e-3):
        super().__init__()

        self.net = net
        self.mask = mask
        self.encoder = encoder
        self.loss = loss
        self.lr = lr

        self.psnr = metrics.PSNR()

    def forward(self, shape: Tuple[int, int, int], T: Tensor):
        c, h, w = shape
        t = T.size(0)
        H = torch.linspace(-1, 1, h).to(T)
        W = torch.linspace(-1, 1, w).to(T)
        gridT, gridH, gridW = torch.meshgrid(T, H, W)
        poses = torch.stack((gridT, gridH, gridW), dim=-1).view(-1, 3)
        encoding = self.encoder(poses)
        if self.training and self.mask:
            encoding *= self.mask(self.global_step).to(encoding)
        return (self.net(encoding).view(t, h, w, c).permute(0, 3, 1, 2) * 0.5 + 0.5)

    def on_train_start(self) -> None:
        print(f'=== learning rate {self.lr} ===')

    def training_step_end(self, outputs):
        self.logger.experiment.add_scalar('Loss', outputs['loss'], self.global_step)
        self.logger.experiment.add_scalar('PSNR', self.psnr(outputs['target'],outputs['predict']), self.global_step)
        return outputs['loss']

    def training_epoch_end(self, outputs):
        dataset = self.trainer.train_dataloader.dataset.datasets
        c,h,w = chw = dataset.videos.shape[-3:]
        T = dataset.T[:8].to(next(self.parameters()).device)
        if self.current_epoch == 1:
            self.logger.experiment.add_video('original', dataset.videos[:8,...].unsqueeze(0), 0)
        if self.current_epoch % 50 == 0:
            dt = T[1] - T[0]
            target = self.forward(chw, T)
            middle = self.forward(chw, T+dt/2)
            combined = torch.stack((target, middle), dim=1).view(-1, *chw)
            self.logger.experiment.add_video('middle', middle.unsqueeze(0), self.current_epoch)
            self.logger.experiment.add_video('target', target.unsqueeze(0), self.current_epoch)
            self.logger.experiment.add_video('combined', combined.unsqueeze(0), self.current_epoch, fps=8)
            del target; del middle; del combined
            self.logger.experiment.add_video('scaled', self.forward((c, h*4, w*4), T[:1]).unsqueeze(0), self.current_epoch)

    def training_step(self, batch, batch_idx):
        video, times = batch
        new_video = self.forward(video.shape[-3:], times)
        loss = self.loss(new_video, video)
        self.log('train_loss', loss)
        return dict(loss=loss, predict=new_video.clamp(0, 1), target=video)

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)

class FlowTrainer(pl.LightningModule):
    def __init__(self, *,
            encoder=nn.Identity(), 
            mask=None,
            net=MLP(2, 3, 256, 3, activation='relu'),
            loss=F.mse_loss,
            lr=1e-3):
        super().__init__()

        self.net = net
        self.mask = mask
        self.encoder = encoder
        self.resample = Resample2d()
        self.loss = loss
        self.lr = lr
        self.data_step = 10

        self.psnr = metrics.PSNR()

    def forward(self, F: Tensor, T: Tensor, factor=-1):
        h, w = F.shape[-2:]
        t = T.size(0)
        H = torch.linspace(-1, 1, h).to(T)
        W = torch.linspace(-1, 1, w).to(T)
        gridT, gridH, gridW = torch.meshgrid(T, H, W)
        poses = torch.stack((gridT, gridH, gridW), dim=-1).view(-1, 3)
        encoding = self.encoder(poses)
        if self.training and self.mask:
            encoding *= self.mask(self.global_step).to(encoding)
        flow = self.net(encoding).view(t, h, w, 2).permute(0, 3, 1, 2) * factor
        pred = self.resample(F.contiguous(), flow.contiguous())
        return pred, flow.norm(dim=1)

    def on_train_start(self) -> None:
        print(f'=== learning rate {self.lr} ===')

    def training_step_end(self, outputs):
        self.logger.experiment.add_scalar('Loss', outputs['loss'], self.global_step)
        self.logger.experiment.add_scalar('PSNR', self.psnr(outputs['target'],outputs['predict']), self.global_step)
        return outputs['loss']

    def training_epoch_end(self, outputs):
        dataset = self.trainer.train_dataloader.dataset.datasets
        self.data_step = dataset.step
    #     T = dataset.T[20:28].to(next(self.parameters()).device)
    #     frames = dataset.videos[20:28,...].to(T.device)
        frames = dataset.videos
    #     chw = frames.shape[-3:]
        if self.current_epoch == 1:
            self.logger.experiment.add_video('original', frames.unsqueeze(0), 0)
    #     if self.current_epoch % 50 == 0:
    #         target, _ = self.forward(frames, T)
    #         middle, flow = self.forward(frames, T, factor=-0.5)
    #         combined = torch.stack((target, middle), dim=1).view(-1, *chw)
    #         combined[::2] = middle
    #         combined[1::2] = target
    #         self.logger.experiment.add_video('outputs', torch.stack((target, middle)), self.current_epoch)
    #         self.logger.experiment.add_video('combined', combined.unsqueeze(0), self.current_epoch)
    #         self.logger.experiment.add_video('flow', flow.unsqueeze(1).unsqueeze(0), self.current_epoch)
    #         del target; del middle; del combined

    def training_step(self, batch, batch_idx):
        frame1, frame2, times = batch
        new_video, _ = self.forward(frame1, times)
        loss = self.loss(new_video, frame2)
        self.log('train_loss', loss)
        return dict(loss=loss, predict=new_video.clamp(0, 1), target=frame1)

    def test_step(self, batch, batch_idx):
        frame1, frame2, times = batch
        chw = frame1.shape[-3:]
        vid_all = [frame1]
        flow_all = []
        for i, t in enumerate(torch.linspace(0, 1, self.data_step + 1)[1:-1]):
            new_video, flow = self.forward(frame1, times, factor=-t)
            vid_all.append(new_video)
            if i == 0:
                flow_all.append(flow)
        vid_all.append(frame2)
        vid_all = torch.stack(vid_all).squeeze()
        flow_all = torch.stack(flow_all).squeeze(1)
        return dict(vid=vid_all, flow=flow_all)

    def test_epoch_end(self, outputs):
        video = torch.cat([seq['vid'] for seq in outputs]).unsqueeze(0)
        flow = torch.stack([seq['flow'] for seq in outputs]).unsqueeze(0)
        self.logger.experiment.add_video('test_video', video, self.current_epoch)
        self.logger.experiment.add_video('test_flow', flow, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        self.test_epoch_end(outputs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)
