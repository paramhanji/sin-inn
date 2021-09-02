from torch import Tensor
import torch
import torch.utils.data as data
import torchvision.io as io
import torchvision
import torchvision.transforms as T
import pytorch_lightning as pl
import os.path as path

torchvision.set_video_backend('pyav')

class VideoClip(data.Dataset):
    def __init__(self, video, start, duration, size=200, step=10):
        super().__init__()
        self.step = step
        self.read_video(video, start, start+duration, step, size)
        self.T = torch.linspace(-1, 1, self.video.size(0))

    def read_video(self, video_path, start, end, step, size):
        self.video, _, infos = io.read_video(video_path, start_pts=start, end_pts=end, pts_unit='sec')
        self.video = self.video[::step]
        self.video = self.video.permute(0, 3, 1, 2).contiguous() / 255
        self.video = T.Resize(size)(self.video)

    def __len__(self):
        return self.video.size(0) - 1

    def __getitem__(self, index):
        return self.video[index], self.video[index+1], self.T[index]

class VideoClipFlow(VideoClip):
    def __init__(self, video, start, duration, size=200, step=10):
        super().__init__(video, start, duration, size, step)
        self.run_raft()

    def run_raft(self, raft_path='/auto/homes/pmh64/projects/RAFT', thresh=1):
        import sys, os, argparse, tqdm
        sys.path.append(os.path.join(raft_path, 'core'))
        from raft import RAFT
        from utils.utils import InputPadder, coords_grid, bilinear_sampler
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help="restore checkpoint",
                            default=os.path.join(raft_path, 'pretrained/raft-things.pth'))
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        args = parser.parse_args([])
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))
        model = model.module
        model.to('cuda')
        model.eval()

        self.flow = []
        with torch.no_grad():
            print('Obtaining optical flow for input frames')
            for im1, im2 in tqdm.tqdm(zip(self.video[:-1], self.video[1:]), total=self.__len__()):
                im1, im2 = (im1*255)[None].to('cuda'), (im2*255)[None].to('cuda')
                padder = InputPadder(im1.shape)
                im1, im2 = padder.pad(im1, im2)
                # _, flow1 = model(im1, im2, iters=20, test_mode=True)
                _, flow2 = model(im2, im1, iters=20, test_mode=True)

                flow2 = padder.unpad(flow2)[0].cpu()
                # coords0 = coords_grid(1, im1.shape[2], im1.shape[3]).to('cuda')
                # coords1 = coords0 + flow1
                # coords2 = coords1 + bilinear_sampler(flow2, coords1.permute(0,2,3,1))
                # err = (coords0 - coords2).norm(dim=1)
                # occ = (err[0] > thresh)
                # flow2 = padder.unpad(flow2 * torch.logical_not(occ[None,None]))[0].cpu()
                self.flow.append(flow2)

        self.flow = torch.stack(self.flow)
        sys.path.pop()

    def __getitem__(self, index):
        return self.video[index], self.video[index+1], self.T[index], self.flow[index]


class VideoModule(pl.LightningDataModule):
    def __init__(self, file: str, start, duration, size=200, step=10, batch=8):
        super().__init__()
        self.trainset = VideoClip(file, start, duration, size=size, step=step)
        self.testset = VideoClipFlow(file, start, duration, size=size, step=step)
        self.batch = batch
    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.trainset, batch_size=self.batch)
    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.testset, batch_size=1)
    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.testset, batch_size=1)
