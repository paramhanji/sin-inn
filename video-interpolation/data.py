import numpy as np, torch
import torch.utils.data as data
import PIL, imageio
import torchvision.transforms as T
import pytorch_lightning as pl
import os, os.path as path
import abc

class BaseMedia(abc.ABC, data.Dataset):
    def __len__(self):
        return self.video.size(0) - 1

    def __getitem__(self, index):
        if self.gt_available:
            return self.video[index], self.video[index+1], self.T[index], self.flow_scale, self.flow[index]
        else:
            return self.video[index], self.video[index+1], self.T[index], self.flow_scale


class VideoClip(BaseMedia):
    def __init__(self, path, start, duration, size=200, step=10):
        super().__init__()
        self.step = step
        self.read_video(path, start, start+duration, step, size)
        trans = T.compose([T.ToTensor(), T.Resize(size)])
        frames = imageio.mimread(path, memtest=False)[start:start+duration:step]
        self.video = trans(frames)
        self.T = torch.linspace(-1, 1, self.video.size(0))
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
                _, flow = model(im1, im2, iters=20, test_mode=True)

                flow = padder.unpad(flow)[0].cpu()
                self.flow.append(flow)

        self.flow = torch.stack(self.flow)
        sys.path.pop()
        self.gt_available = True
        self.flow_scale = 1

class VideoModule(pl.LightningDataModule):
    def __init__(self, file: str, start, duration, size=200, step=10, batch=8):
        super().__init__()
        self.dataset = VideoClip(file, start, duration, size=size, step=step)
        self.batch = batch
    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.dataset, batch_size=self.batch, shuffle=True, num_workers=4)
    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.dataset, batch_size=self.batch, num_workers=4)
    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.dataset, batch_size=self.batch, num_workers=4)


class Images(BaseMedia):
    def __init__(self, root, size=200):
        super().__init__()
        num_frames = len(os.listdir(root))
        frames = [path.join(root, f'frame_{i+1:04d}.png') for i in range(num_frames)]
        w, h = PIL.Image.open(frames[0]).size
        assert h <= w, 'Frame should be landscape oriented'
        trans = T.Compose([lambda x: PIL.Image.open(x), T.ToTensor(), T.Resize(size)])
        self.video = torch.stack([trans(f) for f in frames])
        self.T = torch.linspace(-1, 1, self.video.size(0))

        scene, _ = path.splitext(path.basename(root))
        flow_dir = path.join(root, '../../flow')
        if path.isdir(flow_dir):
            self.gt_available = True
            rescale_ratio = size / h
            flows = [self.readFlow(path.join(flow_dir, scene, f'frame_{i+1:04d}.flo'))
                     for i in range(num_frames - 1)]
            trans = T.Compose([lambda x: torch.tensor(x).permute(2,0,1), T.Resize(size)])
            self.flow = torch.stack([trans(f) for f in flows]) * rescale_ratio
        else:
            self.gt_available = False
        self.flow_scale = self.video.shape[-1] / 5
        # print('Dataset dimensions: ', self.video.shape)

    def readFlow(self, fn):
        '''
        Read .flo file in Middlebury format
        Reference: http://stackoverflow.com/questions/
                   28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

        WARNING: this will work on little-endian architectures (eg Intel x86) only!
        '''
        with open(fn, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                print('Magic number incorrect. Invalid .flo file')
                return None
            else:
                w = np.fromfile(f, np.int32, count=1)
                h = np.fromfile(f, np.int32, count=1)
                # print 'Reading %d x %d flo file\n' % (w, h)
                data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
                # Reshape data into 3D array (columns, rows, bands)
                # The reshape here is for visualization, the original code is (w,h,2)
                return np.resize(data, (int(h), int(w), 2))

class ImagesModule(pl.LightningDataModule):
    def __init__(self, dir, size=200, batch=8):
        super().__init__()
        self.trainset = Images(dir, size=size)
        self.testset = Images(dir, size=436)
        self.batch = batch
    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.trainset, batch_size=self.batch, num_workers=3, shuffle=True)
    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.testset, batch_size=1, num_workers=1)
    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(self.testset, batch_size=1, num_workers=1)


def get_video(input_video, args):
    if path.isdir(path.join(input_video)):
        # print('Loading MPI sintel sequence')
        video_clip = ImagesModule(input_video, size=args.size, batch=args.batch)
    else:
        # print('Extracting frames from video')
        video_clip = VideoModule(input_video, 0, args.end, step=args.step, batch=args.batch, size=args.size)
    scene, _ = path.splitext(path.basename(input_video))
    return video_clip, scene
