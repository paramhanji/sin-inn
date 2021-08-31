from torch import Tensor
import torch
import torch.utils.data as data
import torchvision.io as io
import torchvision
import torchvision.transforms as T
import pytorch_lightning as pl
import os.path as path

torchvision.set_video_backend('pyav')

class FlowImagesData(data.Dataset):
    def __init__(self, dir: str, start, duration, size=200, step=10):
        super().__init__()
        trans = T.Compose([
            lambda x: io.read_image(x) / 255,
            T.Resize(size),
            # T.CenterCrop(size),
            ])
        self.step = step
        self.videos = torch.stack([trans(path.join(dir, f"frame_{i:05d}.png"))
            for i in range(start, start+duration*step, step)], dim=0)
        self.T = torch.linspace(-1, 1, self.videos.size(0))

    def __len__(self):
        return self.videos.size(0) - 1

    def __getitem__(self, index):
        return self.videos[index], self.videos[index+1], self.T[index]


class VideoClip(data.Dataset):
    def __init__(self, video: str, start, duration, size=200, step=10):
        super().__init__()
        self.step = step
        self.videos, _, infos = io.read_video(video, start_pts=start, end_pts=start+duration, pts_unit='sec')
        self.videos = self.videos[::step]
        self.videos = self.videos.permute(0, 3, 1, 2).contiguous().div(255)

        self.videos = T.Resize(size)(self.videos)
        self.T = torch.linspace(-1, 1, self.videos.size(0))
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
            for im1, im2 in tqdm.tqdm(zip(self.videos[:-1], self.videos[1:]), total=self.__len__()):
                im1, im2 = (im1*255)[None].to('cuda'), (im2*255)[None].to('cuda')
                padder = InputPadder(im1.shape)
                im1, im2 = padder.pad(im1, im2)
                _, flow1 = model(im1, im2, iters=20, test_mode=True)
                _, flow2 = model(im2, im1, iters=20, test_mode=True)
                # self.flow.append(flow1)

                coords0 = coords_grid(1, im1.shape[2], im1.shape[3]).cuda()
                coords1 = coords0 + flow1
                coords2 = coords1 + bilinear_sampler(flow2, coords1.permute(0,2,3,1))
                err = (coords0 - coords2).norm(dim=1)
                occ = (err[0] > thresh)
                flow2 = padder.unpad(flow2 * torch.logical_not(occ[None,None]))[0]
                self.flow.append(flow2)

        self.flow = torch.stack(self.flow)
        sys.path.pop()
        # print(self.videos.shape, self.flow.shape)
        # import gfxdisp
        # v = gfxdisp.pfs.pfs()
        # v.view(self.videos[8].permute(1,2,0).cpu().numpy())
        # v.view(self.flow[8].permute(1,2,0).norm(dim=-1).cpu().numpy())
        # exit(-1)


    def __len__(self):
        return self.videos.size(0) - 1

    def __getitem__(self, index):
        return self.videos[index], self.videos[index+1], self.T[index], self.flow[index]


class FlowImagesModule(pl.LightningDataModule):
    def __init__(self, dir: str, start, duration, step, batch=8):
        super().__init__()
        self.dataset = FlowImagesData(dir, start, duration, step=step)
        self.batch = batch
    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.dataset, batch_size=self.batch)
    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.dataset, batch_size=1)

class VideoModule(pl.LightningDataModule):
    def __init__(self, file: str, start, duration, size=200, step=10, batch=8):
        super().__init__()
        self.dataset = VideoClip(file, start, duration, size=size, step=step)
        self.batch = batch
    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.dataset, batch_size=self.batch)
    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.dataset, batch_size=1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    video = VideoClip('data/agent237.mp4', 45.0, 10)
    for frame, time in video:
        plt.imshow(frame.permute(1, 2, 0).numpy())
        plt.show()
