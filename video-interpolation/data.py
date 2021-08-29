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

        # self.videos = T.CenterCrop(size)(self.videos)
        self.videos = T.Resize(size)(self.videos)
        self.T = torch.linspace(-1, 1, self.videos.size(0))
        # self.pfs = infos['video_fps']

    def __len__(self):
        return self.videos.size(0) - 1

    def __getitem__(self, index):
        return self.videos[index], self.videos[index+1], self.T[index]


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
