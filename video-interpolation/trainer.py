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

from my_utils.resample2d import Resample2d
from my_utils.flow_viz import flow2img


class FlowTrainer(pl.LightningModule):
    def __init__(self, *,
            encoder=nn.Identity(), 
            net=None,
            loss=None,
            lr=None,
            step=None,
            flow_scale=None,
            log_gt=True):
        super().__init__()

        self.net = net
        self.encoder = encoder
        self.resample = Resample2d()
        self.loss = loss
        self.lr = lr
        self.data_step = step
        self.flow_scale = flow_scale
        self.write_gt_flow = log_gt

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
        print(f'=== learning rate {self.lr} ===')
        video = self.trainer.train_dataloader.dataset.datasets.videos.unsqueeze(0)
        self.logger.experiment.add_video('original', video, 0)

    def training_step(self, batch, batch_idx):
        frame1, frame2, times, flow = batch
        new_video, new_flow = self.forward(frame1, times)
        loss = self.loss(new_video, frame2)
        # mask = flow != 0
        # loss = ((new_flow[mask] - flow[mask])**2).mean()
        self.log('train_loss', loss)
        return dict(loss=loss, predict=new_video.clamp(0, 1), target=frame2)

    def training_step_end(self, outputs):
        self.logger.experiment.add_scalar('Loss', outputs['loss'], self.global_step)
        self.logger.experiment.add_scalar('PSNR', self.psnr(outputs['target'],outputs['predict']), self.global_step)
        return outputs['loss']

    def validation_step(self, batch, batch_idx):
        frame1, frame2, times, gt_flow = batch
        vid_all, flow_all = [frame1], []
        for i, t in enumerate(torch.linspace(0, 1, self.data_step + 1)[1:-1]):
            new_video, flow = self.forward(frame1, times, factor=t, gt_flow=gt_flow) \
                              if self.write_gt_flow else \
                              self.forward(frame1, times, factor=t)
            vid_all.append(new_video)
            if i == 0:
                flow = torch.cat((flow/t, gt_flow), dim=-1).permute(0,2,3,1)
                flow = flow.squeeze().cpu().numpy().clip(-10, 10)
                flow_image = torch.tensor(flow2img(flow)).unsqueeze(0).permute(0,3,1,2)
                flow_all.append(flow_image)
        vid_all.append(frame2)
        vid_all = torch.stack(vid_all).squeeze()
        flow_all = torch.stack(flow_all).squeeze()
        return dict(vid=vid_all, flow=flow_all)

    def validation_epoch_end(self, outputs):
        video = torch.cat([seq['vid'] for seq in outputs]).unsqueeze(0)
        flow = torch.stack([seq['flow'] for seq in outputs]).unsqueeze(0)
        self.logger.experiment.add_video('val_video', video, self.current_epoch)
        self.logger.experiment.add_video('val_flow', flow, self.current_epoch)
        if self.write_gt_flow:
            print('Logging GT flow')
            self.write_gt_flow = False

    def on_test_start(self):
        print('Start evaluation epoch')
        self.write_gt_flow = False

    def test_step(self, *args):
        return self.validation_step(*args)

    def test_epoch_end(self, outputs):
        video = torch.cat([seq['vid'] for seq in outputs]).permute(0,2,3,1)
        flow = torch.stack([seq['flow'] for seq in outputs]).permute(0,2,3,1)
        io.write_video('interpolated.mp4', video.cpu()*255, fps=self.data_step)
        io.write_video('flow.mp4', flow.cpu()*255, fps=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)
